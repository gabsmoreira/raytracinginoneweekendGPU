#include <iostream>


__global__ void kernel_function(float *pixel, int lenX, int lenY){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index;
    if(i < lenX && j < lenY){
        index = i*3 + j*lenX*3;
        pixel[index + 0] = float(i) / lenX;
        pixel[index + 1] = float(j) / lenY;
        pixel[index + 2] = 0.2;
    }
    else{
        //out of bounds
        return;
    }
}

int main() {
    int nx = 120;
    int ny = 80;
    float *pixel;

    // definindo o tamanho do bloco
    dim3 blocks(nx/8+1,ny/8+1);
    // definindo threads
    dim3 threads(tx,ty);
    // chamando o kernel
    kernel_function<<<blocks, threads>>>(pixel, nx, ny);
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            int index = j*3*nx + i*3;
            float r = pixel[index + 0];
            float g = pixel[index + 1];
            float b = pixel[index + 2];
            int ir = int(255.99*r); 
            int ig = int(255.99*g); 
            int ib = int(255.99*b); 
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}



