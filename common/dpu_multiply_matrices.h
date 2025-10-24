#ifndef __DPU_MULTIPLY_MATRICES_H___
#define __DPU_MULTIPLY_MATRICES_H___

#include "timer.h"

Matrix*  dpu_multiply_matrices(Matrix* matrix1, Matrix* matrix2, uint32_t num_dpus) {
    Timer time;
    // Create a sample matrix multiplication frame
    pim_matrix_multiplication_frame_t* frame = create_pim_matrix_multiplication_frame(num_dpus, 0, matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols, matrix1->rows, matrix2->cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));
    if (!frame) {
        fprintf(stderr, "Frame creation failed");
        return NULL;
    }
    pim_matrix_multiplication_frame_load_first_matrix(frame, matrix1);
    pim_matrix_multiplication_frame_load_second_matrix(frame, matrix2);
    #ifdef TIMER
    startTimer(&time);
    #endif
    pim_matrix_multiplication_frame_execute(frame);
    #ifdef TIMER
    stopTimer(&time);
    printf("DPU multiplication time: %.3f s\n", getElapsedTime(time));
    #endif  
    Matrix* result = pim_matrix_multiplication_frame_get_result(frame);
    if (!result) {
        fprintf(stderr, "Result retrieval failed");
        return NULL;
    }
    pim_matrix_multiplication_frame_free(frame);
    return result;
}

#endif // __DPU_MULTIPLY_MATRICES_H___