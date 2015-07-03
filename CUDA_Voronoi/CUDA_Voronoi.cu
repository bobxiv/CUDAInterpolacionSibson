struct Seed
{
unsigned int x;
unsigned int y;

unsigned int color;
};

//Kernel
__global__ void voronoi(unsigned int* seedsCount, Seed* seeds, size_t* pitch, unsigned int* pixels,int* NX,int* NY)
{
	int x = (blockIdx.x*blockDim.x)+ threadIdx.x;
	int y = (blockIdx.y*blockDim.y)+ threadIdx.y;

	unsigned int Color, distX, distY, tmpDistance;
	unsigned int minDistance=0xFFFFFFFFFFFFFFF;

	__shared__ Seed s[32];

	for(int k=0;k<*seedsCount;k+=32)//leo en mem compartida 32 semillas
	{
		if( threadIdx.y == 0 )
			s[threadIdx.x] = seeds[k + threadIdx.x];

		__syncthreads();//leste todas primero

		#pragma unroll 32
		for(int i=0;i<32;++i)//por cada semilla fetcheada
		{
			distX=s[i].x-x;
			distY=s[i].y-y;

			tmpDistance = distX*distX+distY*distY;

			if( tmpDistance < minDistance )
			{
				Color = s[i].color;
				minDistance= tmpDistance;
			}
		}
		__syncthreads();//termina de usar todas las semillas
	}

	pixels[y*(*pitch)+x]= Color;
	return;
}

#define WINDOWS

#include <iostream>
#ifdef WINDOWS
	#include <windows.h>
	#include <windows.h>
	#include <algorithm>
	#include <string>
#endif
#include <fstream>

//casi 1 millon de pixeles, y con 10000 sitios
#define WIDTH  1280
#define HEIGHT 720

void LoadBitmap(unsigned int**,int&,int&,const std::string);
void SaveBitmap(const unsigned int const*,const std::string);

__host__ int main(int argc, char* argv[])
{
	SetCurrentDirectory("..\\Resources\\");

	//leo la imagen para sacar color de sitios
	unsigned int* imPixels;
	int width=0;
	int height=0;
	LoadBitmap(&imPixels,width,height,"frontier");

		unsigned int w = 112;
		unsigned int h = 100;
		unsigned int h_seedsCount= w*h;
		Seed* h_seeds= new Seed[h_seedsCount];
		
	int blockX = width/w;//112 sitios a lo ancho
	int blockY = height/h;//100 sitios a lo alto
	srand(time(NULL));
	for(int i=1;i<=w;++i)//aca se llenan las semillas|sitios, con colores sacados de la imagen que se carga
		for(int j=1;j<=h;++j)
		{
			unsigned int x= blockX*i-blockX/2;
			unsigned int y= blockY*j-blockY/2;
			h_seeds[(i-1)*h+(j-1)].color =imPixels[y*width+x];//guardo color, de la posicion original
			//pongo el sitio en un rango de desplazamiento del lugar original
			int dx = rand()%(blockX/3+1) * (rand()%2)? 1:-1;
			int dy = rand()%(blockY/3+1) * (rand()%2)? 1:-1;
			h_seeds[(i-1)*h+(j-1)].x=x+dx;
			if( h_seeds[(i-1)*h+(j-1)].x >= WIDTH)
				h_seeds[(i-1)*h+(j-1)].x = WIDTH-1;
			if( h_seeds[(i-1)*h+(j-1)].x < 0)
				h_seeds[(i-1)*h+(j-1)].x = 0;
			h_seeds[(i-1)*h+(j-1)].y=y+dy;
			if( h_seeds[(i-1)*h+(j-1)].y >= HEIGHT)
				h_seeds[(i-1)*h+(j-1)].y = HEIGHT-1;
			if( h_seeds[(i-1)*h+(j-1)].y < 0)
				h_seeds[(i-1)*h+(j-1)].y = 0;
			//h_seeds[(i-1)*100+(j-1)].color =imPixels[(y+dy)*width+(x+dx)];//guardo color
		}
	//for(int k=0;k<(100*100);++k)//esto "saca" algunos sitios para que no sea tan regular
	//	if( rand()%2 )
	//	{
	//		h_seeds[k].color=0;
	//		h_seeds[k].x=0;
	//		h_seeds[k].y=0;
	//	}

	//std::cout<<"blockX: "<<blockX<<std::endl;
	//std::cout<<"blockY: "<<blockY<<std::endl;
	//std::cout<<"blockX/2: "<<blockX/2<<std::endl;
	//std::cout<<"blockY/2: "<<blockY/2<<std::endl;
	//std::cout<<"seeds count: "<<h_seedsCount<<std::endl;
	//for(int y=0;y<h_seedsCount;++y)
	//	std::cout<<"("<<h_seeds[y].x<<";"<<h_seeds[y].y<<") "<<h_seeds[y].color<<' ';
			

	size_t h_pitch=WIDTH;
	//unsigned int h_pixels[]={0, 0, 0, 0,
	//						 0, 0, 0, 0,
	//						 0, 0, 0, 0,
	//						 0, 0, 0, 0
	//						};
	unsigned int* h_pixels=new unsigned int[(long long)WIDTH*(long long)HEIGHT];
	for(int k=0;k<(WIDTH*HEIGHT);++k)
		h_pixels[k]=0;//0x00FFFF;

	int h_NX=WIDTH;//ancho
	int h_NY=HEIGHT;//alto
	//std::cout<<"NX= "<<h_NX<<" NY= "<<h_NY<<std::endl;

	unsigned int* d_seedsCount;
	Seed* d_seeds;
	size_t* d_pitch;
	unsigned int* d_pixels;
	int* NX;
	int* NY;
	cudaMalloc((void**)&d_seedsCount,sizeof(unsigned int));
	cudaMalloc((void**)&d_seeds,sizeof(Seed)*h_seedsCount);
	cudaMalloc((void**)&d_pitch,sizeof(size_t));
	cudaMalloc((void**)&d_pixels,sizeof(unsigned int)*WIDTH*HEIGHT);
	cudaMalloc((void**)&NX,sizeof(int));
	cudaMalloc((void**)&NY,sizeof(int));

	cudaMemcpy(d_seedsCount,&h_seedsCount,sizeof(unsigned int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitch,&h_pitch,sizeof(size_t),cudaMemcpyHostToDevice);
	cudaMemcpy(d_seeds,h_seeds,sizeof(Seed)*h_seedsCount,cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixels,h_pixels,sizeof(unsigned int)*WIDTH*HEIGHT,cudaMemcpyHostToDevice);
	cudaMemcpy(NX,&h_NX,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(NY,&h_NY,sizeof(int),cudaMemcpyHostToDevice);


	//dim3 BlockSize(512,1);
	//dim3 BlockSize(HEIGHT,WIDTH);
	dim3 BlockSize(32,16);
	dim3 GridSize((WIDTH)/32,(HEIGHT)/16);
	LARGE_INTEGER prevTime;
	LARGE_INTEGER curTime;
	LARGE_INTEGER freqTime;
	if( !QueryPerformanceFrequency(&freqTime) )
		exit(1);
	double freqRecip= 1.0/freqTime.QuadPart;
	if( !QueryPerformanceCounter(&prevTime) )
		exit(1);
	//for(int o=0;o<20;++o)
		voronoi<<< GridSize, BlockSize >>>(d_seedsCount,d_seeds,d_pitch,d_pixels,NX,NY);
	
	if( !QueryPerformanceCounter(&curTime) )
		exit(1);
	double ellapsedTime= (curTime.QuadPart-prevTime.QuadPart)*freqRecip;


	std::cout<<"El calculo duro: "<<ellapsedTime<<" s segun timer de Windows"<<std::endl;
	getchar();

	cudaMemcpy(h_pixels,d_pixels,sizeof(unsigned int)*WIDTH*HEIGHT,cudaMemcpyDeviceToHost);

	//Muestra los sitios de voronoi con punto negro en el sitio
	for(int k=0;k<(w*h);++k)
		h_pixels[h_seeds[k].y*WIDTH+h_seeds[k].x] = 0x000000;

	#pragma region salida
		//std::cout<<"Seeds count: "<<h_seedsCount<<std::endl;
		//std::cout<<"Seeds:"<<std::endl;
		//for(int i=0;i < h_seedsCount;++i)
		//	std::cout<<"("<<h_seeds[i].x<<";"<<h_seeds[i].x<<") color "<<h_seeds[i].color<<std::endl;
		//std::cout<<"Pitch: "<<h_pitch<<std::endl;

		//for(int i=0;i < HEIGHT;++i)
		//{
		//	for(int j=0;j < WIDTH;++j)
		//	{
		//		std::cout<<h_pixels[i*4+j]<<' ';
		//	}
		//	std::cout<<std::endl;
		//}
	#pragma endregion 

	SaveBitmap(h_pixels,"Image");

	delete[] h_pixels;
	//delete[] h_seeds;
	cudaFree(d_seedsCount);
	cudaFree(d_seeds);
	cudaFree(d_pitch);
	cudaFree(d_pixels);

	system("Image.bmp");

	return 0;
}

struct BGR//en bitmap se guarda bgr
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
};

void LoadBitmap(unsigned int** pixels,int& width, int& height, const std::string Name)
{
	std::ifstream Input((Name+".bmp").c_str(), std::ios::binary);
	if( !Input.is_open() )
		exit(1);
	int offset=0;
	Input.seekg(10,std::ios::cur);
	Input.read((char*)&offset,4);

	Input.seekg(4,std::ios::cur);
	Input.read((char*)&width,4);
	Input.read((char*)&height,4);

	unsigned short bitsPerPixel=0;
	unsigned int compressionType=0;
	Input.seekg(2,std::ios::cur);
	Input.read((char*)&bitsPerPixel,2);
	Input.read((char*)&compressionType,4);
	if( compressionType != 0 )//solo leo sin compression
		abort();
	if( bitsPerPixel != 24 )//nomas leo de 24 bits
		abort();
	//std::cout<<"bits per pixel: "<<bitsPerPixel<<" compression: "<<compressionType<<std::endl;

	Input.seekg(offset,std::ios::beg);

	BGR* colors=new BGR[width*height];
	for(int k=0;k <(width*height);++k)
		Input.read((char*)&(colors[k]),sizeof(BGR));

	*pixels=new unsigned int[width*height];
	for(int k=0;k <(width*height);++k)
		(*pixels)[k]= RGB(colors[k].b,colors[k].g,colors[k].r);//lo guardo como BGR
	//Input.read((char*)*pixels,width*height*sizeof(int));

	Input.close();
	
	std::reverse(*pixels,*pixels+width*height);
	for(int y=0;y<height;++y)
		std::reverse(*pixels+y*width,*pixels+(y+1)*width);

	return;
}

void SaveBitmap(const unsigned int const* pixels,const std::string Name)
{
	unsigned int* my_pixels=new unsigned int[(long long)WIDTH*(long long)HEIGHT];
	std::copy(pixels,pixels+WIDTH*HEIGHT,my_pixels);

	std::reverse(my_pixels,my_pixels+WIDTH*HEIGHT);
	for(int i=0;i < HEIGHT;++i)
	{
		std::reverse(my_pixels+i*WIDTH,my_pixels+i*WIDTH+WIDTH);
	}

	BITMAPFILEHEADER BMHeader;
	BMHeader.bfType= 0x4D42;//MB -> swap -> BM			77 d-> 4D hex	66 d -> 42 hex
	BMHeader.bfReserved1=0;
	BMHeader.bfReserved2=0;
	BMHeader.bfSize= sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+4*4*sizeof(RGBQUAD);
	BMHeader.bfOffBits= sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER);
	
	BITMAPINFO BMInfo;

	BITMAPINFOHEADER BMInfoHeader;
	ZeroMemory(&BMInfoHeader,sizeof(BMInfoHeader));
	BMInfoHeader.biBitCount= 32;
	BMInfoHeader.biPlanes=1;
	BMInfoHeader.biSize = sizeof(BMInfoHeader);
	BMInfoHeader.biHeight = HEIGHT;
	BMInfoHeader.biWidth = WIDTH;
	BMInfoHeader.biCompression=BI_RGB;
	BMInfoHeader.biSizeImage= 0;

	BMInfo.bmiHeader=BMInfoHeader;
	//BMInfo.bmiColors=h_pixels;

	HANDLE Image =  CreateFile((Name+".bmp").c_str(),GENERIC_WRITE,NULL,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);

	if( !Image )
		exit(1);

	int byteW=0;
	WriteFile(Image,(const void*)&BMHeader,sizeof(BITMAPFILEHEADER),(LPDWORD)&byteW,NULL);

	WriteFile(Image,(const void*)&BMInfo.bmiHeader,sizeof(BITMAPINFOHEADER),(LPDWORD)&byteW,NULL);

	WriteFile(Image,(const void*)my_pixels,sizeof(RGBQUAD)*WIDTH*HEIGHT,(LPDWORD)&byteW,NULL);

	CloseHandle(Image);
	delete[] my_pixels;
	return;
}



