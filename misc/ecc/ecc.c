#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint8_t bit_parity(int input) {
    uint8_t par = 0;
    while(input > 0) {
        par = !par;
        input &= (input - 1);
    }

    return par;
}

void ecc_encode(uint8_t input_bits[], int len, uint8_t output_bits[]) {
    const uint8_t poly1 = 7;
    const uint8_t poly2 = 5;

    uint8_t working_mem = 0;
    uint8_t parity1[len];
    uint8_t parity2[len];

    for(int i = 0; i < len; i++) {
        working_mem = ((working_mem << 1) + input_bits[i]) & 0x07;

        parity1[i] = bit_parity(working_mem & poly1);
        parity2[i] = bit_parity(working_mem & poly2);
    }

    for(int i = 0; i < len; i++) {
        output_bits[i] = parity1[i];
        output_bits[len + i] = parity2[i];
    }
}

int main() {
    uint8_t input_bytes[] = {15, 24, 68, 253};
    const int input_bytes_len = sizeof(input_bytes);

    /*uint8_t input_bits[] = {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0};*/
    const int input_bits_len = 8 * sizeof(input_bytes);
    uint8_t *input_bits = malloc(input_bits_len);

    printf("size: %lu, %lu, %d, %lu\n", (sizeof(input_bits) / sizeof(input_bits[0])), sizeof(input_bits), input_bytes_len, sizeof(input_bytes));

    for(int i = 0; i < input_bytes_len; i++) {
        for(int j = 0; j < 8; j++) {
            input_bits[8*i + j] = ((input_bytes[i] >> (7-j)) & 1);
        }
    }

    uint8_t output_bits[2 * input_bits_len];
    ecc_encode(input_bits, input_bits_len, output_bits);

    printf("Input:          ");
    for(int i = 0; i < input_bits_len; i++) {
        printf("%u", input_bits[i]);

        if( (i+1) % 8 == 0) {
            printf("_");
        }
    }
    printf("\n");

    printf("Output:         ");
    for(int i = 0; i < 2*input_bits_len; i++) {
        printf("%u", output_bits[i]);

        if( (i+1) % 8 == 0) {
            printf("_");
        }
    }
    printf("\n");

    printf("Output (bytes): ");
    for(int i = 0; i < 2*input_bytes_len; i++) {
        uint8_t cur_byte = 0;
        for(int j = 0; j < 8; j++) {
            cur_byte += output_bits[8*i + j] << (7-j);
        }
        printf("%u,", cur_byte);
    }

    printf("\n");
}
