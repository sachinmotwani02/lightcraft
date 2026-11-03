//! portra-looks fork: bake a photo's current look into a film LUT (a RAW-log `.cube`, see [`crate::lut::film_shaper`]).
//!
//! Each grid point u is the film LUTs' own input: linear sRGB after the camera exposure,
//! lin = 0.18 · (2^(15u − 9) − 2^−12). The grid runs through the render's own per-pixel stage
//! ([`crate::finish::finish_with`]) as a small image, so the bake and the render can't drift apart. Baked: white
//! balance, Exposure, calibration, the film LUT at its Amount (below 100 % blended with LightCraft's tone map for this
//! photo), colour (saturation, vibrance, mixer, point colour, grading), curves, sRGB encode. Not baked (they need
//! neighbours or a position): Highlights/Shadows, Texture, Clarity, Dehaze, sharpening, noise reduction, vignette,
//! grain, masks, spots, lens corrections; [`not_baked`] lists the ones in use.

use lightcraft_develop::DevelopSettings;
use lightcraft_raster::{Plane, Rgb32f};
use std::sync::Arc;

use crate::finish::{FinishParams, FinishParamsInput, finish_with};
use crate::{OutputSpace, Prepared, SourceInfo};

/// Linear sRGB (after the camera exposure) at RAW-log `u`: the inverse of [`crate::lut::film_shaper`].
pub fn film_input(u: f32) -> f32 {
    0.18 * ((15.0 * u - 9.0).exp2() - 2f32.powi(-12))
}

/// `s` on a source with `info` as a `size`³ film LUT: sRGB-encoded entries (0..1) in `.cube` order (red fastest).
pub fn bake_film(info: &SourceInfo, s: &DevelopSettings, size: usize) -> Vec<[f32; 3]> {
    let size = size.max(2);
    // the settings the render uses (section eyes, SDR rendition, built-in profile sliders)
    let effective = s.effective();
    let sdr = effective.sdr_rendition();
    let s = crate::profiles::effective(&sdr);
    let (w, h) = (size * size, size);
    let from_709 = lightcraft_color::SRGB.to_space(&lightcraft_color::REC2020).to_f32();
    // the render feeds the film LUT scene × the photo's camera exposure; divide it out, so the LUT sees u itself
    let k = info.film_ev.exp2();
    let step = 1.0 / (size - 1) as f32;
    let mut img = Rgb32f::new(w, h);
    for (i, px) in img.data.iter_mut().enumerate() {
        let lin = [i % size, i / size % size, i / (size * size)].map(|q| film_input(q as f32 * step) / k);
        *px = std::array::from_fn(|r| (0..3).map(|c| from_709[r][c] * lin[c]).sum());
    }
    crate::local::white_balance(&mut img, info, &s);
    let frame = crate::geometry::Frame::new(w, h, &s, false);
    let mut fp = FinishParams::new(FinishParamsInput {
        settings: &s,
        frame: &frame,
        info,
        image_size: (w, h),
        px_per_long: 1.0,
        air_pre: 0.0,
        space: OutputSpace::Srgb,
    });
    // spatial and local stages off (with no masks and no blur planes, local tone adds nothing)
    (fp.vig, fp.grain) = (None, None);
    (fp.hl, fp.sh, fp.clar, fp.tex, fp.dehaze, fp.sharpen, fp.air) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let prep = Prepared {
        img: Arc::new(img),
        log_l: Arc::new(Plane::new(w, h)),
        base: Arc::new(Plane::new(w, h)),
        clarity_blur: None,
        texture_blur: None,
        dark: None,
        chroma_blur: None,
        air: 0.0,
        masks: Vec::new(),
        px_per_long: 1.0,
    };
    finish_with(&prep, &fp, true, |e| e.map(|v| v.clamp(0.0, 1.0)))
}

/// The settings in use (not at their defaults: a raw's own for raws) that a bake leaves out.
pub fn not_baked(s: &DevelopSettings, info: &SourceInfo) -> Vec<&'static str> {
    let s = &*s.effective();
    let d = if info.raw { DevelopSettings::for_raw(0.0, 0.0) } else { DevelopSettings::default() };
    let effects = s.section_enabled("effects");
    let optics = s.section_enabled("optics");
    [
        ("Highlights", s.light.highlights != d.light.highlights),
        ("Shadows", s.light.shadows != d.light.shadows),
        ("Texture", effects && s.effects.texture != d.effects.texture),
        ("Clarity", effects && s.effects.clarity != d.effects.clarity),
        ("Dehaze", effects && s.effects.dehaze != d.effects.dehaze),
        ("Vignette", effects && s.vignette.amount != d.vignette.amount),
        ("Grain", effects && s.grain.amount != d.grain.amount),
        ("Sharpening", s.detail.sharpen_amount != d.detail.sharpen_amount),
        ("Noise reduction", s.detail.nr_luminance != d.detail.nr_luminance || s.detail.nr_color != d.detail.nr_color),
        ("Denoise", s.enhance != d.enhance),
        ("Lens corrections", optics && s.optics != d.optics),
        ("Lens blur", s.lens_blur != d.lens_blur),
        ("Camera local tone", s.profile.camera_local_tone != d.profile.camera_local_tone),
        ("Masks", !s.masks.is_empty()),
        ("Spots", !s.spots.is_empty()),
        ("Red eye", !s.red_eye.is_empty()),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect()
}

/// A `.cube` file of `entries` (`size`³, as [`bake_film`] gives them).
pub fn to_cube(title: &str, size: usize, entries: &[[f32; 3]]) -> String {
    let mut out = format!("TITLE \"{}\"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n", title.replace('"', "'"));
    for e in entries {
        out += &format!("{:.6} {:.6} {:.6}\n", e[0], e[1], e[2]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut::{Lut3d, film_shaper, register};

    #[test]
    fn film_input_inverts_the_shaper() {
        for i in 0..=64 {
            let u = i as f32 / 64.0;
            assert!((film_shaper([film_input(u); 3])[0] - u).abs() < 2e-6, "{u}");
        }
    }

    #[test]
    fn untouched_bake_gives_back_the_film_lut() {
        // a film LUT with some colour in it, at 9³ so the test stays quick
        let n = 9;
        let mut text = String::from("TITLE \"t: RAW-log in\"\nLUT_3D_SIZE 9\n");
        for i in 0..n * n * n {
            let [r, g, b] = [i % n, i / n % n, i / (n * n)].map(|q| q as f32 / (n - 1) as f32);
            text += &format!("{} {} {}\n", (r * 0.8 + g * 0.2).powf(0.9), g * 0.9 + 0.05 * b, (b * b + r * 0.1).min(1.0));
        }
        let lut = Lut3d::parse_cube(&text).unwrap();
        register("lut:test-bake", lut.clone());
        let mut s = DevelopSettings::default();
        s.profile.id = "lut:test-bake".into();
        s.profile.amount = 100.0;
        let info = SourceInfo { raw: true, film_ev: 0.7, ..SourceInfo::default() };
        let baked = bake_film(&info, &s, n);
        let worst = baked.iter().zip(&lut.data).flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs())).fold(0.0, f32::max);
        assert!(worst < 1e-3, "worst {worst}");
        assert_eq!(not_baked(&s, &info), ["Sharpening", "Noise reduction"]);
        let cube = Lut3d::parse_cube(&to_cube("x: RAW-log in", n, &baked)).unwrap();
        assert!(cube.raw_log && cube.size == n);
    }
}
