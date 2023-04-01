//  8 BIT LINEAR-QUANTIZATION
//  https://onnxruntime.ai/docs/performance/quantization.html#quantization-overview

use array::ArrayTrait;
use option::OptionTrait;
use neural_network::onnx_cairo::operators::math::int33;
use neural_network::onnx_cairo::operators::math::int33::i33;
use neural_network::onnx_cairo::operators::math::int33::max;
use neural_network::onnx_cairo::operators::math::int33::abs;
use neural_network::onnx_cairo::operators::math::vector::find_min;
use neural_network::onnx_cairo::operators::math::vector::find_max;


fn symetric_quant(min_val: i33, max_val: i33, data: i33) -> i33 {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    //  Define quantization range
    //  int8 range : [-127;127] 
    let q_min_int = i33 { inner: 127_u32, sign: true };
    let q_max_int = i33 { inner: 127_u32, sign: false };

    let factor = i33 { inner: 1000_u32, sign: false };
    let min_val = min_val * factor;
    let max_val = max_val * factor;

    //  Calculate the scale based on 8 bit symetric quantization
    //  scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)
    let scale = (max(abs(min_val), abs(max_val)) * i33 { inner: 2_u32, sign: false })
        / (q_max_int - q_min_int);

    //  Quantize data based on the scale
    let quantized_data = (data * factor) / scale;

    assert(quantized_data.inner <= 127_u32, 'out of range');

    return quantized_data;
}

fn quant_vec(vec: @Array::<i33>) -> Array::<i33> {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    let mut result = ArrayTrait::<i33>::new();

    let mut min_val = find_min(vec);
    let mut max_val = find_max(vec);

    __quant_vec(ref min_val, ref max_val, vec, ref result, 0_usize);

    return result;
}

fn __quant_vec(
    ref min_val: i33, ref max_val: i33, vec: @Array::<i33>, ref result: Array::<i33>, n: usize
) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    // --- End of the recursion ---
    if n == vec.len() {
        return ();
    }

    // --- Quantize data and append to the result vector --- 
    let quantized = symetric_quant(min_val, max_val, *vec.at(n));
    result.append(quantized);

    // --- The process is repeated for the remaining elemets in the array --- 
    __quant_vec(ref min_val, ref max_val, vec, ref result, n + 1_usize)
}
