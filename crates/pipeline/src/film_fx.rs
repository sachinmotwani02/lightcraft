//! portra-looks film softness and film grain (sachinmotwani02 fork): two spatial passes over the
//! finished, sRGB-encoded image, after a film LUT profile (a `.cube` whose TITLE says "RAW-log in").
//! Ports of `portra-looks/code/soft.py` (preset "raw", or "jpeg" for non-raw input) and `grain.py`.
//! Off by default (`profile.film_soft` 0, `profile.film_grain` 0); controls `film.soft`, `film.grain`.
//!
//! Sizes are in px at 1024 px long edge, scaled by the whole output's long edge / 1024 (as in the
//! Python), so a preview, a zoomed window and a 12 MP export get the same look. Grain is white noise
//! hashed from full-output pixel coordinates, so a window shows the same grain as the whole render;
//! softness reads neighbours, so windowed renders are grown by [`margin`] (see `plan`).

use lightcraft_develop::DevelopSettings;

use crate::for_rows;

/// soft.py presets (sizes at 1024 px long edge, except `dehalo`: native px).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Soft {
    dehalo: f32,
    sigma_y: f32,
    sigma_c: f32,
    veil: f32,
    sigma_veil: f32,
}

/// `soft.py` "raw": the RAW decode is halo-free, so tone is left alone.
const SOFT_RAW: Soft = Soft { dehalo: 0.0, sigma_y: 0.0, sigma_c: 1.3, veil: 0.08, sigma_veil: 2.5 };
/// `soft.py` "jpeg": camera-processed (sharpened) input.
const SOFT_JPEG: Soft = Soft { dehalo: 1.0, sigma_y: 0.55, sigma_c: 1.3, veil: 0.10, sigma_veil: 2.5 };

/// A grain.py strength: L* += amount·shape(L*)·N(size); a*, b* += amount·chroma·shape(L*)·N(size·csize).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Grain {
    amount: f32,
    size: f32,
    chroma: f32,
    csize: f32,
    /// shape(L*) at L* = [`KX`].
    knots: [f32; 7],
}

const KX: [f32; 7] = [0.0, 10.0, 25.0, 50.0, 70.0, 85.0, 100.0];

/// `film.grain` 1, 2, 3: subtle, fine, strong. The grain.py in git has only fine (2.5) and strong (4.0);
/// these are fitted by simulation to the renders Sachin's newer grain.py made, R2
/// `own_raw/film/golden_vivid_roll_grain-<strength>/` (L* std per band within 1 % from L* 10 up, colour
/// std within 3 %; `portra-looks/code/lc_grain_fit.py`). Subtle and fine keep the shadows light; strong
/// has grain.py's original shape.
const GRAINS: [Grain; 3] = [
    Grain { amount: 0.98, size: 0.5, chroma: 0.35, csize: 1.5, knots: [0.19, 0.6, 1.0, 0.93, 0.65, 0.38, 0.19] },
    Grain { amount: 1.94, size: 0.5, chroma: 0.27, csize: 1.5, knots: [0.18, 0.58, 1.0, 0.93, 0.65, 0.37, 0.19] },
    Grain { amount: 3.98, size: 0.5, chroma: 0.28, csize: 1.9, knots: [0.44, 0.71, 0.99, 1.0, 0.69, 0.36, 0.18] },
];

/// The film stages of one render.
#[derive(Clone, Debug, PartialEq)]
pub struct FilmFx {
    /// Softness blend 0..1 (1 = the soft.py preset).
    soft: f32,
    preset: Soft,
    /// The de-halo blur in output px (soft.py: 1 native px).
    dehalo_px: f32,
    grain: Option<Grain>,
    seed: u32,
    /// Whole output long edge / 1024.
    sc: f32,
    /// This render's offset in the whole output (px).
    origin: (i64, i64),
}

/// Whether `s` has a film LUT profile with a film stage on.
fn active(s: &DevelopSettings) -> bool {
    (s.profile.film_soft > 0.0 || s.profile.film_grain >= 0.5) && crate::lut::get(&s.profile.id).is_some_and(|l| l.raw_log)
}

/// Pixels the softness reads beyond a window of a `full_long` px output (0 when off).
pub fn margin(s: &DevelopSettings, full_long: usize) -> usize {
    if !active(s) || s.profile.film_soft <= 0.0 {
        return 0;
    }
    let sc = full_long as f32 / 1024.0;
    let p = SOFT_JPEG; // the wider of the two presets
    (4.0 * p.sigma_veil.max(p.sigma_c).max(p.sigma_y) * sc + 4.0 * p.dehalo + 3.0).ceil() as usize
}

/// `win` grown by [`margin`] on every side, inside the `full_w × full_h` output.
pub fn grow(s: &DevelopSettings, full_w: usize, full_h: usize, win: crate::PixelWindow) -> crate::PixelWindow {
    let m = margin(s, full_w.max(full_h));
    if m == 0 {
        return win;
    }
    let (x, y) = (win.x.saturating_sub(m), win.y.saturating_sub(m));
    crate::PixelWindow { x, y, w: (win.x + win.w + m).min(full_w) - x, h: (win.y + win.h + m).min(full_h) - y }
}

impl FilmFx {
    /// The film stages of a `w × h` render of `s` in `frame` (a window when `frame.view` is set):
    /// `None` unless the profile is a film LUT and a stage is on. SDR only (`peak` 1).
    pub fn new(
        s: &DevelopSettings,
        info: &crate::SourceInfo,
        frame: &crate::geometry::Frame,
        (w, h): (usize, usize),
        px_per_long: f64,
        peak: f32,
    ) -> Option<FilmFx> {
        if !active(s) || peak > 1.0 {
            return None;
        }
        let view = frame.view.map_or([0.0, 0.0, w as f32, h as f32], |v| [v.x as f32, v.y as f32, v.full_w as f32, v.full_h as f32]);
        // output px per px of the original (soft.py's de-halo is in native px)
        let out_per_native = px_per_long / info.native_long.unwrap_or(0).max(frame.src_w.max(frame.src_h)).max(1) as f64;
        let raw = info.raw;
        let level = s.profile.film_grain.round().clamp(0.0, 3.0) as usize;
        let full_long = view[2].max(view[3]).max(1.0);
        Some(FilmFx {
            soft: (s.profile.film_soft / 100.0).clamp(0.0, 1.0) as f32,
            preset: if raw { SOFT_RAW } else { SOFT_JPEG },
            dehalo_px: (out_per_native as f32).min(4.0),
            grain: (level > 0).then(|| GRAINS[level - 1]),
            seed: s.grain.seed,
            sc: full_long / 1024.0,
            origin: (view[0].round() as i64, view[1].round() as i64),
        })
    }

    /// Soften, then grain, `img` (`w × h`, sRGB-encoded, in place; clipped to 0..1 as in the Python).
    pub fn apply(&self, img: &mut [[f32; 3]], w: usize, h: usize) {
        if w == 0 || h == 0 {
            return;
        }
        if self.soft > 0.0 {
            soften(img, w, h, &self.preset, self.sc, self.dehalo_px, self.soft);
        }
        if let Some(g) = &self.grain {
            add_grain(img, w, h, g, self.sc, self.origin, self.seed);
        }
    }
}

// ---------- softness (soft.py `soften`) ----------

const YCC: [[f32; 3]; 3] = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]];
const YCC_INV: [[f32; 3]; 3] = [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]];

fn mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn soften(img: &mut [[f32; 3]], w: usize, h: usize, p: &Soft, sc: f32, dehalo_px: f32, amount: f32) {
    let ycc: Vec<[f32; 3]> = img.iter().map(|&c| mul(&YCC, c)).collect();
    let plane = |k: usize| -> Vec<f32> { ycc.iter().map(|v| v[k]).collect() };
    let mut y = plane(0);
    if p.dehalo > 0.0 {
        // clamp Y to the 5 × 5 min..max of Y blurred by `dehalo` native px (cuts sharpening rims)
        let g = gauss(&y, w, h, p.dehalo * dehalo_px);
        let (lo, hi) = (rank5(&g, w, h, f32::min), rank5(&g, w, h, f32::max));
        for i in 0..y.len() {
            y[i] = y[i].clamp(lo[i], hi[i].max(lo[i]));
        }
    }
    if p.sigma_y > 0.0 {
        y = gauss(&y, w, h, p.sigma_y * sc);
    }
    if p.veil > 0.0 {
        // highlight veil: Y mixes toward a wide blur, weight veil · smoothstep(0.55, 0.95, Y)
        let b = gauss(&y, w, h, p.sigma_veil * sc);
        for (v, b) in y.iter_mut().zip(&b) {
            let t = ((*v - 0.55) / 0.40).clamp(0.0, 1.0);
            let k = p.veil * t * t * (3.0 - 2.0 * t);
            *v = *v * (1.0 - k) + b * k;
        }
    }
    let (cb, cr) =
        if p.sigma_c > 0.0 { (gauss(&plane(1), w, h, p.sigma_c * sc), gauss(&plane(2), w, h, p.sigma_c * sc)) } else { (plane(1), plane(2)) };
    for (i, c) in img.iter_mut().enumerate() {
        let s = mul(&YCC_INV, [y[i], cb[i], cr[i]]).map(|v| v.clamp(0.0, 1.0));
        *c = std::array::from_fn(|k| c[k] + (s[k] - c[k]) * amount);
    }
}

/// scipy's Gaussian kernel: radius `int(4σ + 0.5)`, normalized.
fn kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma + 0.5) as i64;
    let k: Vec<f32> = (-r..=r).map(|x| (-0.5 * (x as f32 / sigma).powi(2)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.into_iter().map(|v| v / s).collect()
}

/// scipy's "reflect" border (… b a | a b …).
#[inline]
fn reflect(i: i64, n: i64) -> usize {
    let p = 2 * n;
    let m = i.rem_euclid(p);
    (if m < n { m } else { p - 1 - m }) as usize
}

/// Gaussian blur of a `w × h` plane (separable; scipy `gaussian_filter`, mode "reflect").
fn gauss(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let k = kernel(sigma);
    let r = (k.len() / 2) as i64;
    let mut tmp = vec![0.0f32; w * h];
    for_rows(&mut tmp, w, |yy, row| {
        let s = &src[yy * w..(yy + 1) * w];
        // the row with its reflected borders, then a plain dot product per pixel
        let line: Vec<f32> = (-r..w as i64 + r).map(|x| s[reflect(x, w as i64)]).collect();
        for (x, o) in row.iter_mut().enumerate() {
            *o = k.iter().zip(&line[x..x + k.len()]).map(|(a, b)| a * b).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    for_rows(&mut out, w, |yy, row| {
        for (j, kv) in k.iter().enumerate() {
            let sy = reflect(yy as i64 + j as i64 - r, h as i64);
            for (o, v) in row.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *o += kv * v;
            }
        }
    });
    out
}

/// 5 × 5 minimum or maximum (scipy `minimum_filter`/`maximum_filter`, size 5, mode "reflect").
fn rank5(src: &[f32], w: usize, h: usize, f: fn(f32, f32) -> f32) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    for_rows(&mut tmp, w, |yy, row| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = (-2..=2).map(|d| src[yy * w + reflect(x as i64 + d, w as i64)]).reduce(f).unwrap_or(0.0);
        }
    });
    let mut out = vec![0.0f32; w * h];
    for_rows(&mut out, w, |yy, row| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = (-2..=2).map(|d| tmp[reflect(yy as i64 + d, h as i64) * w + x]).reduce(f).unwrap_or(0.0);
        }
    });
    out
}

// ---------- grain (grain.py `add_grain`) ----------

/// Unit Gaussian white noise at whole-output pixel (x, y) of field `f` (hash + Box–Muller).
#[inline]
fn white(x: i64, y: i64, seed: u32, f: u32) -> f32 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ ((seed as u64) << 32 | f as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    // splitmix64 finalizer
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u1 = ((z >> 40) as f32 + 1.0) / 16_777_217.0; // (0, 1]
    let u2 = ((z & 0xFF_FFFF) as f32) / 16_777_216.0;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// A `w × h` unit-std noise field: white noise blurred by `sigma` px (none at ≤ 0.3, as grain.py),
/// for the window at `origin` of the whole output. The same output pixel gets the same value in
/// any window.
fn field(w: usize, h: usize, sigma: f32, origin: (i64, i64), seed: u32, f: u32) -> Vec<f32> {
    let (ox, oy) = origin;
    if sigma <= 0.3 {
        let mut out = vec![0.0f32; w * h];
        for_rows(&mut out, w, |y, row| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = white(ox + x as i64, oy + y as i64, seed, f);
            }
        });
        return out;
    }
    let k = kernel(sigma);
    let r = k.len() / 2;
    // blurred white noise has std √Σk² per axis: Σk² for the 2D separable blur
    let norm = 1.0 / k.iter().map(|v| v * v).sum::<f32>();
    let wr = w + 2 * r;
    // horizontal pass over the rows the vertical pass reads (h + 2r of them)
    let mut tmp = vec![0.0f32; w * (h + 2 * r)];
    for_rows(&mut tmp, w, |yy, row| {
        let gy = oy + yy as i64 - r as i64;
        let line: Vec<f32> = (0..wr).map(|x| white(ox + x as i64 - r as i64, gy, seed, f)).collect();
        for (x, o) in row.iter_mut().enumerate() {
            *o = k.iter().zip(&line[x..x + k.len()]).map(|(a, b)| a * b).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    for_rows(&mut out, w, |yy, row| {
        for (j, kv) in k.iter().enumerate() {
            let s = &tmp[(yy + j) * w..(yy + j + 1) * w];
            for (o, v) in row.iter_mut().zip(s) {
                *o += kv * v;
            }
        }
        for o in row.iter_mut() {
            *o *= norm;
        }
    });
    out
}

fn add_grain(img: &mut [[f32; 3]], w: usize, h: usize, g: &Grain, sc: f32, origin: (i64, i64), seed: u32) {
    let nl = field(w, h, g.size * sc, origin, seed, 0);
    let na = field(w, h, g.size * g.csize * sc, origin, seed, 1);
    let nb = field(w, h, g.size * g.csize * sc, origin, seed, 2);
    for_rows(img, w, |y, row| {
        for (x, c) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let mut lab = rgb2lab(*c);
            let k = interp(lab[0], &g.knots) * g.amount;
            lab[0] += k * nl[i];
            lab[1] += k * g.chroma * na[i];
            lab[2] += k * g.chroma * nb[i];
            *c = lab2rgb(lab);
        }
    });
}

/// `np.interp(l, KX, knots)`.
fn interp(l: f32, knots: &[f32; 7]) -> f32 {
    if l <= KX[0] {
        return knots[0];
    }
    for i in 1..KX.len() {
        if l <= KX[i] {
            let t = (l - KX[i - 1]) / (KX[i] - KX[i - 1]);
            return knots[i - 1] + (knots[i] - knots[i - 1]) * t;
        }
    }
    knots[6]
}

// CIELAB of sRGB-encoded colour, D65 (portra-looks `pl.rgb2lab` / `pl.lab2rgb`; clips to 0..1)
const M: [[f32; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
const MI: [[f32; 3]; 3] =
    [[3.240_454_2, -1.537_138_5, -0.498_531_4], [-0.969_266_0, 1.876_010_8, 0.041_556_0], [0.055_643_4, -0.204_025_9, 1.057_225_2]];
const WHITE: [f32; 3] = [0.95047, 1.0, 1.08883];
const EPS: f32 = 216.0 / 24389.0;
const KAP: f32 = 24389.0 / 27.0;

fn rgb2lab(c: [f32; 3]) -> [f32; 3] {
    use lightcraft_color::transfer::srgb_to_linear;
    let xyz = mul(&M, c.map(|v| srgb_to_linear(v.clamp(0.0, 1.0))));
    let f: [f32; 3] = std::array::from_fn(|i| {
        let t = xyz[i] / WHITE[i];
        if t > EPS { t.max(1e-12).cbrt() } else { (KAP * t + 16.0) / 116.0 }
    });
    [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])]
}

fn lab2rgb(lab: [f32; 3]) -> [f32; 3] {
    use lightcraft_color::transfer::linear_to_srgb;
    let fy = (lab[0] + 16.0) / 116.0;
    let (fx, fz) = (fy + lab[1] / 500.0, fy - lab[2] / 200.0);
    let finv = |f: f32| if f * f * f > EPS { f * f * f } else { (116.0 * f - 16.0) / KAP };
    let yy = if lab[0] > KAP * EPS { fy * fy * fy } else { lab[0] / KAP };
    let xyz = [finv(fx) * WHITE[0], yy * WHITE[1], finv(fz) * WHITE[2]];
    mul(&MI, xyz).map(|v| linear_to_srgb(v.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fx(soft: f32, grain: usize, sc: f32, origin: (i64, i64)) -> FilmFx {
        FilmFx { soft, preset: SOFT_RAW, dehalo_px: 1.0, grain: (grain > 0).then(|| GRAINS[grain - 1]), seed: 7, sc, origin }
    }

    fn test_image(w: usize, h: usize) -> Vec<[f32; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                [0.5 + 0.4 * (x * 0.21).sin(), 0.45 + 0.4 * (y * 0.13).cos(), if (x as usize / 9 + y as usize / 7) % 2 == 0 { 0.9 } else { 0.2 }]
            })
            .collect()
    }

    #[test]
    fn off_without_a_film_lut() {
        let mut s = DevelopSettings::default();
        s.profile.film_soft = 100.0;
        s.profile.film_grain = 2.0;
        let frame = crate::geometry::Frame::new(100, 80, &s, true);
        let info = crate::SourceInfo { raw: true, ..Default::default() };
        assert!(FilmFx::new(&s, &info, &frame, (100, 80), 100.0, 1.0).is_none(), "no film LUT profile: no film stages");
        assert_eq!(margin(&s, 4032), 0);
    }

    #[test]
    fn lab_round_trip() {
        for c in [[0.2, 0.5, 0.8], [0.9, 0.1, 0.4], [0.5, 0.5, 0.5]] {
            let back = lab2rgb(rgb2lab(c));
            assert!(c.iter().zip(back).all(|(a, b)| (a - b).abs() < 1e-4), "{c:?} -> {back:?}");
        }
        assert!((rgb2lab([1.0; 3])[0] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn blur_keeps_flat_and_mean() {
        let (w, h) = (37, 23);
        let flat = vec![0.3f32; w * h];
        assert!(gauss(&flat, w, h, 2.5).iter().all(|v| (v - 0.3).abs() < 1e-6));
        let k = kernel(1.3);
        assert_eq!(k.len(), 2 * 5 + 1);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!((reflect(-1, 5), reflect(-2, 5), reflect(5, 5), reflect(6, 5)), (0, 1, 4, 3));
    }

    #[test]
    fn grain_has_its_amount_in_the_mids() {
        // mid grey (L* ≈ 53, shape 0.9..1): luma std ≈ amount · shape
        let (w, h) = (400, 300);
        for (level, amount) in [(1, 1.0), (2, 2.0), (3, 4.0)] {
            let mut img = vec![[0.5f32; 3]; w * h];
            fx(0.0, level, 1.0, (0, 0)).apply(&mut img, w, h);
            let l: Vec<f32> = img.iter().map(|&c| rgb2lab(c)[0]).collect();
            let m = l.iter().sum::<f32>() / l.len() as f32;
            let sd = (l.iter().map(|v| (v - m).powi(2)).sum::<f32>() / l.len() as f32).sqrt();
            let want = amount * interp(53.4, &GRAINS[level - 1].knots);
            assert!((sd / want - 1.0).abs() < 0.08, "level {level}: std {sd}, want {want}");
        }
    }

    #[test]
    fn a_window_matches_the_whole_render() {
        let (w, h) = (160, 120);
        let full = test_image(w, h);
        let f = fx(1.0, 2, 0.6, (0, 0));
        let mut whole = full.clone();
        f.apply(&mut whole, w, h);
        // a window with a margin wide enough for the softness: its middle is the whole render's
        let (x0, y0, ww, wh, m) = (50usize, 30usize, 60usize, 50usize, 12usize);
        let mut win: Vec<[f32; 3]> = (0..wh).flat_map(|y| full[(y0 + y) * w + x0..(y0 + y) * w + x0 + ww].to_vec()).collect();
        fx(1.0, 2, 0.6, (x0 as i64, y0 as i64)).apply(&mut win, ww, wh);
        for y in m..wh - m {
            for x in m..ww - m {
                let (a, b) = (win[y * ww + x], whole[(y0 + y) * w + x0 + x]);
                assert!(a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-5), "({x}, {y}): {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn softness_blurs_colour_not_tone() {
        let (w, h) = (64, 64);
        let mut img = test_image(w, h);
        let before = img.clone();
        fx(1.0, 0, 2.0, (0, 0)).apply(&mut img, w, h);
        let y = |c: [f32; 3]| mul(&YCC, c)[0];
        // the raw preset leaves Y alone below the veil's range
        for (a, b) in before.iter().zip(&img) {
            if y(*a) < 0.5 {
                assert!((y(*a) - y(*b)).abs() < 2e-3, "{a:?} {b:?}");
            }
        }
        assert!(before.iter().zip(&img).any(|(a, b)| (a[2] - b[2]).abs() > 0.05), "chroma blurred");
    }
}
