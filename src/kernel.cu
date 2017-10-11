/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../../cuda_by_example/common/book.h"
#include "../../cuda_by_example/common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) {
		return r*r + i*i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale*(float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale*(float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a*a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;

}

__global__ void kernel(unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y*gridDim.x;

	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;

}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	dim3 grid(DIM, DIM);

    kernel<<<grid, 1>>>(dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_bitmap));

	bitmap.display_and_exit();

}
*/
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda.h"
#include "../../cuda_by_example/common/book.h"
#include "../../cuda_by_example/common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// these exist on the GPU side
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;



// this kernel takes in a 2-d array of floats
// it updates the value-of-interest by a scaled value based
// on itself and its nearest neighbors
__global__ void blend_kernel(float *dst,
	bool dstOut) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == DIM - 1) right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)   top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	float   t, l, c, r, b;
	if (dstOut) {
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);

	}
	else {
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// NOTE - texOffsetConstSrc could either be passed as a
// parameter to this function, or passed in __constant__ memory
// if we declared it as a global above, it would be
// a parameter here: 
// __global__ void copy_const_kernel( float *iptr,
//                                    size_t texOffset )
__global__ void copy_const_kernel(float *iptr) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0)
		iptr[offset] = c;
}

// globals needed by the update routine
struct DataBlock {
	unsigned char   *output_bitmap;
	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;
	CPUAnimBitmap  *bitmap;

	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;
};

void anim_gpu(DataBlock *d, int ticks) {
	HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3    blocks(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	CPUAnimBitmap  *bitmap = d->bitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i<90; i++) {
		float   *in, *out;
		if (dstOut) {
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else {
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		}
		copy_const_kernel << <blocks, threads >> >(in);
		blend_kernel << <blocks, threads >> >(out, dstOut);
		dstOut = !dstOut;
	}
	float_to_color << <blocks, threads >> >(d->output_bitmap,
		d->dev_inSrc);

	HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
		d->output_bitmap,
		bitmap->image_size(),
		cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(d->stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		d->start, d->stop));
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame:  %3.1f ms\n",
		d->totalTime / d->frames);
}

// clean up memory allocated on the GPU
void anim_exit(DataBlock *d) {
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);
	HANDLE_ERROR(cudaFree(d->dev_inSrc));
	HANDLE_ERROR(cudaFree(d->dev_outSrc));
	HANDLE_ERROR(cudaFree(d->dev_constSrc));

	HANDLE_ERROR(cudaEventDestroy(d->start));
	HANDLE_ERROR(cudaEventDestroy(d->stop));
}


int main(void) {
	DataBlock   data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	HANDLE_ERROR(cudaEventCreate(&data.start));
	HANDLE_ERROR(cudaEventCreate(&data.stop));

	int imageSize = bitmap.image_size();

	HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap,
		imageSize));

	// assume float == 4 chars in size (ie rgba)
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc,
		imageSize));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc,
		imageSize));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc,
		imageSize));

	HANDLE_ERROR(cudaBindTexture(NULL, texConstSrc,
		data.dev_constSrc,
		imageSize));

	HANDLE_ERROR(cudaBindTexture(NULL, texIn,
		data.dev_inSrc,
		imageSize));

	HANDLE_ERROR(cudaBindTexture(NULL, texOut,
		data.dev_outSrc,
		imageSize));

	// intialize the constant data
	float *temp = (float*)malloc(imageSize);
	for (int i = 0; i<DIM*DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800; y<900; y++) {
		for (int x = 400; x<500; x++) {
			temp[x + y*DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp,
		imageSize,
		cudaMemcpyHostToDevice));

	// initialize the input data
	for (int y = 800; y<DIM; y++) {
		for (int x = 0; x<200; x++) {
			temp[x + y*DIM] = MAX_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp,
		imageSize,
		cudaMemcpyHostToDevice));
	free(temp);

	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu,
		(void(*)(void*))anim_exit);
}
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "../../cuda_by_example/common/book.h"
#include "../../cuda_by_example/common/cpu_bitmap.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include <math.h>

PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;

#define     DIM    512

GLuint  bufferObj;
cudaGraphicsResource *resource;

// based on ripple code, but uses uchar4 which is the type of data
// graphic inter op uses. see screenshot - basic2.png
__global__ void kernel(uchar4 *ptr) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char   green = 128 + 127 *
		sin(abs(fx * 100) - abs(fy * 100));

	// accessing uchar4 vs unsigned char*
	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

static void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		// clean up OpenGL and CUDA
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

static void draw_func(void) {
	// we pass zero as the last parameter, because out bufferObj is now
	// the source, and the field switches from being a pointer to a
	// bitmap to now mean an offset into a bitmap object
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}


int main(int argc, char **argv) {
	cudaDeviceProp  prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

	// tell CUDA which dev we will be using for graphic interop
	// from the programming guide:  Interoperability with OpenGL
	//     requires that the CUDA device be specified by
	//     cudaGLSetGLDevice() before any other runtime calls.

	HANDLE_ERROR(cudaGLSetGLDevice(dev));

	// these GLUT calls need to be made before the other OpenGL
	// calls, else we get a seg fault
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("bitmap");

	glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

	// the first three are standard OpenGL, the 4th is the CUDA reg 
	// of the bitmap these calls exist starting in OpenGL 1.5
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4,
		NULL, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(
		cudaGraphicsGLRegisterBuffer(&resource,
			bufferObj,
			cudaGraphicsMapFlagsNone));

	// do work with the memory dst being on the GPU, gotten via mapping
	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	uchar4* devPtr;
	size_t  size;
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr,
			&size,
			resource));

	dim3    grids(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	kernel <<<grids, threads >>>(devPtr);
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

	// set up GLUT and kick off main loop
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();
}
