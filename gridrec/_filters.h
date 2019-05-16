#ifndef __FILTERS_H__
#define __FILTERS_H__

void calc_filter(float *filter, unsigned long nang, unsigned long N , float center , int type_filter , int radon_degree);
float convolv_nn(float x , float *lut);
float convolv_lin(float x , float *lut);

#endif __FILTERS_H__
