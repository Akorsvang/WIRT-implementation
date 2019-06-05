//TODO: Use the bitvec crate

fn print_bits(arr: &[u8]) {
    for i in 0..arr.len() {
        print!("{:08b}_", arr[i]);
    }
    println!("");
}

fn bit_parity(value: u8) -> u8 {
    let mut par = false;
    let mut value_mut = value;

    while value_mut > 0 {
        par = !par;
        value_mut &= value_mut - 1;
    }

    par as u8
}

fn conv_encode(input: &[u8]) -> Vec<u8> {
    const POLY1: u8 = 7;
    const POLY2: u8 = 5;

    let mut working_mem = 0;
    let mut parity1 : Vec<u8> = Vec::with_capacity(2 * 8 * input.len());
    let mut parity2 : Vec<u8> = Vec::with_capacity(2 * 8 * input.len());


    for i in 0..input.len() {
        let (mut par_byte1, mut par_byte2) = (0u8, 0u8);

        for j in 0..8 {
            working_mem = ( (working_mem << 1) + ((input[i] >> (7-j)) & 1u8) ) & 0x07u8;

            par_byte1 += bit_parity(working_mem & POLY1) << (7 - j);
            par_byte2 += bit_parity(working_mem & POLY2) << (7 - j);
        }

        parity1.push(par_byte1);
        parity2.push(par_byte2);
    }

    parity1.append(&mut parity2);
    parity1
}

fn main() {
    const INPUT_BYTES: [u8; 4] = [15, 24, 68, 253];
    println!("Input: {:?}", INPUT_BYTES);

    for i in 0..INPUT_BYTES.len() {
        for j in 0..8 {
            print!("{:?}", ((INPUT_BYTES[i] >> (7 - j)) & 0x01));
        }
        print!("_")
    }
    println!("");

    print_bits(&INPUT_BYTES);

    let result = conv_encode(&INPUT_BYTES);

    print_bits(&result);

    print!("Output (bytes): ");
    for i in 0..result.len() {
        print!("{:?},",  result[i]);
    }
    println!("");

}