#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"

__device__ vec3 color(const ray& r, hitable *world) {
    hit_record rec;
    if (world->hit(r, 0.0, MAXFLOAT, rec)) {
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void kernel_function(float *pixels, int lenX, int lenY, hitable *world, hitable **list){
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // int index;
    // vec3 lower_left_corner(-2.0, -1.0, -1.0);
    // vec3 horizontal(4.0, 0.0, 0.0);
    // vec3 vertical(0.0, 2.0, 0.0);
    // vec3 origin(0.0, 0.0, 0.0);
    printf("hehe\n");
    // if((i >= lenX) || (j >= lenY)) return;

    // float u = float(i) / float(lenX);
    // float v = float(j) / float(lenY);
    // ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    // vec3 col = color(r, world);
    
    // index = i*3 + j*lenX*3;
    // pixels[index + 0] = col[0];
    // pixels[index + 1] = col[1];
    // pixels[index + 2] = col[2];
}

__global__ void kernel_init(hitable **list, hitable *world){
    list[0] = new sphere(vec3(0,0,-1), 0.5);
    list[1] = new sphere(vec3(0,-100.5,-1), 100);
    world = new hitable_list(list, 2);
}

int main() {
    int nx = 120;
    int ny = 80;
    int num_pixels = nx*ny;
    size_t size_pixels = 3*num_pixels*sizeof(float);
    size_t size_list = 2*sizeof(hitable);
    size_t size_world = sizeof(hitable_list);
    
    float *pixels;
    float *pixelsCPU;
    hitable *list;
    hitable *world;
    cudaMallocManaged((void **)&pixels, size_pixels);
    cudaMallocManaged((void **)&list, size_list);
    cudaMallocManaged((void **)&world, size_world);

    // definindo o tamanho do bloco
    dim3 blocks(nx/8+1,ny/8+1);

    // definindo threads
    dim3 threads(8,8);

   

    // chamando o kernel init para criar hitable list e world
    kernel_init<<<1, 1>>>(&list, world);
    printf("ola");

    // sincronizar kernels
    cudaDeviceSynchronize();

    // chamando a funcao que calcula os pixels
    kernel_function<<<blocks, threads>>>(pixels, nx, ny, world, &list);

    // memcopy do pixel device -> host
    cudaMemcpy(pixelsCPU, pixels, size_pixels, cudaMemcpyDeviceToHost);

    cudaGetLastError();
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    // vec3 lower_left_corner(-2.0, -1.0, -1.0);
    // vec3 horizontal(4.0, 0.0, 0.0);
    // vec3 vertial(0.0, 2.0, 0.0);
    // vec3 origin(0.0, 0.0, 0.0);
    // list[0] = new sphere(vec3(0,0,-1), 0.5);
    // list[1] = new sphere(vec3(0,-100.5,-1), 100);
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            int index = j*3*nx + i*3;
            float r = pixelsCPU[index + 0];
            float g = pixelsCPU[index + 1];
            float b = pixelsCPU[index + 2];
            int ir = int(255.99*r); 
            int ig = int(255.99*g); 
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}
