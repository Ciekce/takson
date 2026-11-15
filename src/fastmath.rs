const EXP_BIAS: i32 = f32::MAX_EXP - 1;
const MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
const MANTISSA_MASK: u32 = (1 << MANTISSA_BITS) - 1;

#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-40.0, 40.0);
    let k = (x * std::f32::consts::LOG2_E).round();
    let r = (-k).mul_add(std::f32::consts::LN_2, x);

    let mut p: f32 = 1.0 / 120.0;
    for c in [1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = p.mul_add(r, c);
    }

    let scale = f32::from_bits(((k as i32 + EXP_BIAS) << 23) as u32);
    p * scale
}

#[allow(unused)]
#[inline(always)]
pub fn log(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> MANTISSA_BITS) as i32 - EXP_BIAS) as f32;
    let mut m = f32::from_bits((bits & MANTISSA_MASK) | (EXP_BIAS << MANTISSA_BITS) as u32);
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1.0;
    }

    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut t: f32 = 1.0 / 9.0;
    for c in [1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0] {
        t = t.mul_add(s2, c);
    }
    e.mul_add(std::f32::consts::LN_2, 2.0 * s * t)
}
