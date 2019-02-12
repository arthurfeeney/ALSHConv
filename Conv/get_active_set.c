
#include <TH/TH.h>
#include <stdio.h> // PRINT SOME ERROR THING??? PLEASE!!!

int row_hist(THLongTensor* hist, THLongTensor* tensor, int64_t nbins) 
{
    /*
     * Computes histogram for each row of input tensor. 
     * output into hist.
     * min is 0 and max is nbins assumed. 
     */
    
    int64_t nrows = THLongTensor_size(tensor, 0);
    int64_t row_len = THLongTensor_size(tensor, 1);

    //THLongTensor_resize2d(hist, nrows, nbins);
    THLongTensor_zero(hist);

    for(int64_t row = 0; row < nrows; ++row) {
        for(int64_t i = 0; i < row_len; ++i) {
            int64_t bin = THLongTensor_data(tensor)[row*row_len + i];
            ++hist->storage->data[row*nbins + (bin)];
        }
    }

    return 1;
}

int k_freq_buckets(THLongTensor* ti, // rows are hashes of each tensor
                   int64_t k,
                   int64_t bins, 
                   THLongTensor* output) 
{
    int64_t nrows = THLongTensor_size(ti, 0);

    // find frequencies of each element. 
    THLongTensor* hist = THLongTensor_newWithSize2d(nrows, bins);
    row_hist(hist, ti, bins);

    THLongTensor_resize2d(output, nrows, k);

    THLongTensor* values = THLongTensor_newWithSize2d(nrows, k);

    // put indices of top k most frequent elements of hist into output.
    for(int64_t row = 0; row < nrows; ++row) {
        for(int64_t i = 0; i < bins; ++i) {
            THLongTensor_topk(values, output, hist, k, 1, 1, 0);
        }
    }
    

    free(hist);
    free(values);
    return 1;
}         






