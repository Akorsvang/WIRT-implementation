#include <time.h>
#include <fftw3.h>

#define MAX_ITER 10000000ul
#define N (625)

int main()
{
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE |  FFTW_DESTROY_INPUT);

    struct timespec time1, time2;
    /*clock_gettime(CLOCK_MONOTONIC, &time1);*/
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

    for (int i = 0; i < MAX_ITER; i++)
    {
        fftw_execute(p);
    }

    /*clock_gettime(CLOCK_MONOTONIC, &time2);*/
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

    unsigned long int time_sec = time2.tv_sec  - time1.tv_sec;
    long int time_ns  = time2.tv_nsec - time1.tv_nsec;

    printf("%lu s + %ld ns\n", time_sec, time_ns);
    printf("Time taken: %lf ns per iteration\n", ((1e9 * time_sec) + (double) time_ns)/MAX_ITER);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}
