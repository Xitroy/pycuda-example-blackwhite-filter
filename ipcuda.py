from PIL import Image
import time
 
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy 

def CudaBlackWhite(inPath , outPath):
    
    #Timer
    totalT0 = time.perf_counter()
    
    #Open image. Image is an array
    im = Image.open(inPath)
    px = numpy.array(im)
    px = px.astype(numpy.float32)
 
    #Allocate necessary amount of memory. Copy array data to gpu memory
    d_px = cuda.mem_alloc(px.nbytes)
    cuda.memcpy_htod(d_px, px)
 
    #Kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024,1,1)
    checkSize = numpy.int32(im.size[0]*im.size[1])
    grid = (int(im.size[0]*im.size[1]/BLOCK_SIZE)+1,1,1)
 
    #Kernel text
    kernel = '''
    __global__ void bw( float *inIm, int check ){
 
        int idx = (threadIdx.x ) + blockDim.x * blockIdx.x ;
        if(idx *3 < check*3)
        {
       		int val = 0.21 *inIm[idx*3] + 0.71*inIm[idx*3+1] + 0.07 * inIm[idx*3+2];
        	inIm[idx*3]= val;
        	inIm[idx*3+1]= val;
        	inIm[idx*3+2]= val;
        }
    }
    '''
    
    #Compile and get kernel function
    mod = SourceModule(kernel)
    func = mod.get_function("bw")
    func(d_px,checkSize, block=block,grid = grid)
 
    #Get back data from gpu
    bwPx = numpy.empty_like(px)
    cuda.memcpy_dtoh(bwPx, d_px)
    bwPx = (numpy.uint8(bwPx))
 
    #Save image
    pil_im = Image.fromarray(bwPx,mode ="RGB")
    pil_im.save(outPath)
     
    #Time result
    totalT1 = time.perf_counter()
    totalTime = totalT1-totalT0
 
    print ("Black and white image")
    print ("Image size: ",im.size)
    print ("Total execution time : " ,totalTime)

#Run function
CudaBlackWhite('4K.jpg','4Kb.jpg')
