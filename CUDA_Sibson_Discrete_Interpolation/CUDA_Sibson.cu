//#define DEBUG
#include "CUDA_Sibson.h" 

//Devices & Kernels functions
	//__global__ void sibson(const unsigned int const*, const Seed const*, const size_t const*, Pixel*, const unsigned int const*, P*);
	//__device__ void voronoi(const unsigned int const*,const Seed const*, const size_t const*, Pixel*);


//device function 
//Requiere que la cantidad de semillas sean multiplo de 32
//Parametros:
// IN		seedCount cantidad de semillas en array seeds
// IN		seeds     arreglo de semillas en la grilla
// IN		pitch     ancho de una fila
// IN/OUT	pixels    array de pixeles
__device__ void voronoi(const unsigned int const* seedsCount, Seed * seeds, const size_t const* pitch, Pixel* pixels)
{
	int x = (blockIdx.x*blockDim.x)+ threadIdx.x;
	int y = (blockIdx.y*blockDim.y)+ threadIdx.y;

	unsigned int distX, distY, tmpDistance;
	unsigned char r, g, b; 
	int ClosestSeedIndex;
	unsigned int minDistance= FAR; 

	__shared__ Seed s[4/*32*/];

	for(int k=0;k<*seedsCount;k+=4/*32*/)//lectura de a 32 semillas
	{
		if( threadIdx.y == 0 )
			s[threadIdx.x] = seeds[k + threadIdx.x];

		__syncthreads();

		#pragma unroll 4/*32*/
		for(int i=0;i<4/*32*/;++i)//por cada semilla fetcheada
		{
			distX=s[i].x-x;
			distY=s[i].y-y;

			tmpDistance = distX*distX+distY*distY;

			if( tmpDistance < minDistance )
			{
				r = s[i].r;
				g = s[i].g;
				b = s[i].b;
				minDistance= tmpDistance;
				ClosestSeedIndex=i;
			}
		}
		__syncthreads();//termina de usar todas las semillas
	}

	pixels[y*(*pitch)+x].r= r;
	pixels[y*(*pitch)+x].g= g;
	pixels[y*(*pitch)+x].b= b;
	pixels[y*(*pitch)+x].SeedClosest= &seeds[ClosestSeedIndex];
	pixels[y*(*pitch)+x].SeedDst= minDistance;
	return;
}

//__device__ void voronoi(const unsigned int const*,const Seed const*, const size_t const*, Pixel*);
//Kernel
//Parametros:
// IN		seedCount cantidad de semillas en array seeds
// IN		seeds     arreglo de semillas en la grilla
// IN		pitch     ancho de una fila
// IN/OUT	pixels    array de pixeles
// IN		pCount    cantidad de puntos en array p
// IN/OUT	p         array de puntos a interpolar
__global__ void sibson(const unsigned int const* seedsCount,  Seed * seeds, const size_t const* pitch, Pixel* pixels, const unsigned int const* pCount, P* p)
{
	
	voronoi(seedsCount,seeds,pitch,pixels);//cargo el voronoi de los pixeles

	__syncthreads();

	int x = (blockIdx.x*blockDim.x)+ threadIdx.x;
	int y = (blockIdx.y*blockDim.y)+ threadIdx.y;
		//printf("\nPixel (%i;%i)",x,y);
	unsigned int r = pixels[y*(*pitch)+x].SeedDst;
	Seed* ClosestSeed = pixels[y*(*pitch)+x].SeedClosest;

	//__shared__ P Ptmp[32];

	for(int k=0;k<*pCount;++k)//por todo pixel p
	{

			//printf("\n\nk = %i",k);
			//printf("\np: (%i;%i)",p[k].x,p[k].y);
		unsigned int dx=p[k].x-x;
		unsigned int dy=p[k].y-y;
			//printf("\ndx = %i    dy = %i    r = %i    dst = %i",dx,dy,r,(dx*dx+dy*dy));
		if( (dx*dx+dy*dy) <= r /*&& (dx*dx+dy*dy) > 0*/ )//dentro de esfera
		{
				//printf("\nDentro de Esfera! con dst: %i",(dx*dx+dy*dy));
			atomicAdd((unsigned int*)&(p[k].r),(unsigned int)ClosestSeed->r);
			atomicAdd((unsigned int*)&(p[k].g),(unsigned int)ClosestSeed->g);
			atomicAdd((unsigned int*)&(p[k].b),(unsigned int)ClosestSeed->b);
			atomicAdd((unsigned int*)&(p[k].n),1);
			//Ptmp[k].r += ClosestSeed.r;
			//Ptmp[k].g += ClosestSeed.g;
			//Ptmp[k].b += ClosestSeed.b;
			//Ptmp[k].n++;
				//printf("\nClosestSeed (%i;%i) R = %i - G = %i - B = %i",ClosestSeed.x,ClosestSeed.y,ClosestSeed.r,ClosestSeed.g,ClosestSeed.b);
				//printf(" n = %i",p[k].n);
		}
	}

	return;
}




__host__ int main(int argc, char* argv[])
{
	//CUT_DEVICE_INIT();
	//unsigned int timer;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
		//carpeta de recursos
	SetCurrentDirectory("..\\Resources\\");

		//leo la imagen para sacar colores de sitios
	unsigned int* imPixels;
	unsigned int width=0;
	unsigned int height=0;
	LoadBitmap(&imPixels,width,height,"Soft");

	unsigned int h_seedsCount= 4;
	Seed* h_seeds= new Seed[h_seedsCount];

	#pragma region Inicializar las semillas

	h_seeds[0].r= 255;h_seeds[0].g= 0;  h_seeds[0].b= 0;  h_seeds[0].x=(WIDTH/4.0)*1; h_seeds[0].y=(HEIGHT/4.0)*1;
	h_seeds[1].r= 255;h_seeds[1].g= 255;h_seeds[1].b= 0;  h_seeds[1].x=(WIDTH/4.0)*3; h_seeds[1].y=(HEIGHT/4.0)*1;
	h_seeds[2].r= 0;  h_seeds[2].g= 0;  h_seeds[2].b= 255;h_seeds[2].x=(WIDTH/4.0)*1; h_seeds[2].y=(HEIGHT/4.0)*3;
	h_seeds[3].r= 0;  h_seeds[3].g= 255;h_seeds[3].b= 0;  h_seeds[3].x=(WIDTH/4.0)*3; h_seeds[3].y=(HEIGHT/4.0)*3;
	/*
		int blockX = width/112;//112 sitios a lo ancho
		int blockY = height/100;//100 sitios a lo alto
		srand(time(NULL));
		for(int i=1;i<=112;++i)//aca se llenan las semillas|sitios, con colores sacados de la imagen que se carga
			for(int j=1;j<=100;++j)
			{
				unsigned int x= blockX*i-blockX/2;
				unsigned int y= blockY*j-blockY/2;
				h_seeds[(i-1)*100+(j-1)].color =imPixels[y*width+x];//guardo color, de la posicion original
				//pongo el sitio en un rango de desplazamiento del lugar original
				int dx = rand()%(blockX/3+1) * (rand()%2)? 1:-1;
				int dy = rand()%(blockY/3+1) * (rand()%2)? 1:-1;
				h_seeds[(i-1)*100+(j-1)].x=x+dx;
				if( h_seeds[(i-1)*100+(j-1)].x >= WIDTH)
					h_seeds[(i-1)*100+(j-1)].x = WIDTH-1;
				if( h_seeds[(i-1)*100+(j-1)].x < 0)
					h_seeds[(i-1)*100+(j-1)].x = 0;
				h_seeds[(i-1)*100+(j-1)].y=y+dy;
				if( h_seeds[(i-1)*100+(j-1)].y >= HEIGHT)
					h_seeds[(i-1)*100+(j-1)].y = HEIGHT-1;
				if( h_seeds[(i-1)*100+(j-1)].y < 0)
					h_seeds[(i-1)*100+(j-1)].y = 0;
				//h_seeds[(i-1)*100+(j-1)].color =imPixels[(y+dy)*width+(x+dx)];//guardo color
			}
		//for(int k=0;k<(100*100);++k)//esto "saca" algunos sitios para que no sea tan regular
		//	if( rand()%2 )
		//	{
		//		h_seeds[k].color=0;
		//		h_seeds[k].x=0;
		//		h_seeds[k].y=0;
		//	}
		*/

	#ifdef MY_DEBUG
		//std::cout<<"blockX: "<<blockX<<std::endl;
		//std::cout<<"blockY: "<<blockY<<std::endl;
		//std::cout<<"blockX/2: "<<blockX/2<<std::endl;
		//std::cout<<"blockY/2: "<<blockY/2<<std::endl;
		//std::cout<<"seeds count: "<<h_seedsCount<<std::endl;
		//for(int y=0;y<h_seedsCount;++y)
		//	std::cout<<"("<<h_seeds[y].x<<";"<<h_seeds[y].y<<") c: "<<h_seeds[y].r<<' '
		//	<<h_seeds[y].g<<' '<<h_seeds[y].b;
	#endif

	#pragma endregion

	size_t h_pitch=WIDTH;
	Pixel* h_pixels=new Pixel[WIDTH*HEIGHT];

	#pragma region Inicializar los pixeles
		for(int k=0;k<(WIDTH*HEIGHT);++k)
		{
			h_pixels[k].r=0;
			h_pixels[k].g=0;
			h_pixels[k].b=0;
			h_pixels[k].SeedDst= FAR;
			//h_pixels[k].SeedClosest=NULL;
		}
	#pragma endregion

	unsigned int h_pCount=(WIDTH*HEIGHT);//-h_seedsCount;
	P* h_p=new P[h_pCount];

	#pragma region Inicializar los puntos a interpolar(p)
		for(int k=0;k<h_pCount;++k)
		{
			h_p[k].r=0;
			h_p[k].g=0;
			h_p[k].b=0;
			h_p[k].n=0;
			h_p[k].x=k%h_pitch;
			h_p[k].y=k/h_pitch;
		}
	#pragma endregion


		//variables en device
	unsigned int* d_seedsCount;
	Seed* d_seeds;
	size_t* d_pitch;
	Pixel* d_pixels;
	unsigned int* d_pCount;
	P* d_p;

	cudaMalloc((void**)&d_seedsCount,sizeof(unsigned int));
	cudaMalloc((void**)&d_seeds,sizeof(Seed)*h_seedsCount);
	cudaMalloc((void**)&d_pitch,sizeof(size_t));
	cudaMalloc((void**)&d_pixels,sizeof(Pixel)*WIDTH*HEIGHT);
	cudaMalloc((void**)&d_pCount,sizeof(unsigned int));
	cudaMalloc((void**)&d_p,sizeof(P)*h_pCount);

	cudaMemcpy(d_seedsCount,&h_seedsCount,sizeof(unsigned int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitch,&h_pitch,sizeof(size_t),cudaMemcpyHostToDevice);
	cudaMemcpy(d_seeds,h_seeds,sizeof(Seed)*h_seedsCount,cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixels,h_pixels,sizeof(Pixel)*WIDTH*HEIGHT,cudaMemcpyHostToDevice);
	cudaMemcpy(d_pCount,&h_pCount,sizeof(unsigned int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_p,h_p,sizeof(P)*h_pCount,cudaMemcpyHostToDevice);


	dim3 BlockSize(32,16);
	dim3 GridSize(WIDTH/32,HEIGHT/16);

	LARGE_INTEGER prevTime;
	LARGE_INTEGER curTime;
	LARGE_INTEGER freqTime;
	if( !QueryPerformanceFrequency(&freqTime) )
		abort();
	double freqRecip= 1.0/freqTime.QuadPart;

	//CUT_SAFE_CALL(cutStartTimer(timer));

	if( !QueryPerformanceCounter(&prevTime) )
		abort();

		sibson<<< GridSize, BlockSize >>>(d_seedsCount,d_seeds,d_pitch,d_pixels,d_pCount,d_p); 
		//voronoi<<< GridSize, BlockSize >>>(d_seedsCount,d_seeds,d_pitch,d_pixels,NX,NY);
	
	if( !QueryPerformanceCounter(&curTime) )
		exit(1);
	double ellapsedTime= (curTime.QuadPart-prevTime.QuadPart)*freqRecip;

	//CUT_SAFE_CALL(cutStopTimer(timer));

	std::cout<<"El calculo duro: "<<ellapsedTime<<" s segun timer de Windows\n"<<std::endl;
	std::cout<<"El calculo duro: "<</*cutGetTimerValue()/1000.0*/" ----- "<<" s segun timer de CUDA\n";
	getchar();

		//sacando los resultados
	cudaMemcpy(h_pixels,d_pixels,sizeof(Pixel)*WIDTH*HEIGHT,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p,d_p,sizeof(P)*h_pCount,cudaMemcpyDeviceToHost);

	//#ifdef MY_DEBUG
	//std::cout<<"Detalle de los pixeles de voronoi:\n";
	//	for(int k=0;k<(WIDTH*HEIGHT);++k)
	//	{
	//		std::cout<<"|("<<(h_pixels[k].SeedClosest).x<<';'<<(h_pixels[k].SeedClosest).y<<") dst: "<<h_pixels[k].SeedDst<<"| ";
	//		if( ((k+1)%h_pitch) == 0 )
	//			std::cout<<'\n';
	//	}
	//#endif

	for(int k=0;k<h_pCount;++k)//normalizar los valores interpolados
	{
		#ifdef MY_DEBUG
			std::cout<<"n: "<<h_p[k].n<<" x: "<<h_p[k].x<<" y: "<<h_p[k].y;
			std::cout<<" - c: "<<(unsigned int)(unsigned char)h_p[k].r<<" "<<(unsigned int)(unsigned char)h_p[k].g<<" "<<(unsigned int)(unsigned char)h_p[k].b;
			if( h_p[k].n == 0 )
			{
				std::cout<<'\n';
			}
			std::cout<<" - c/n: "<<h_p[k].r/h_p[k].n<<" "<<h_p[k].g/h_p[k].n<<" "<<h_p[k].b/h_p[k].n<<" \n";
		#endif
		h_p[k].r = h_p[k].r/h_p[k].n;
		h_p[k].g = h_p[k].g/h_p[k].n;
		h_p[k].b = h_p[k].b/h_p[k].n;
		h_p[k].n = 0;
	}

	for(int k=0;k<h_pCount;++k)//guardo en los pixels los resultados de h_p, es decir los valores interpolados
	{						   //en el lugar de los pixeles, para poder verlos
		for(int h=0;h<(WIDTH*HEIGHT);++h)
		{
			if( (h_p[k].y*h_pitch+h_p[k].x)==h )
			{
			h_pixels[h].r=h_p[k].r;
			h_pixels[h].g=h_p[k].g;
			h_pixels[h].b=h_p[k].b;
			}
		}
	}

		//paso de la estructura Pixel a un arreglo de usigned int
	unsigned int* res_pixels= new unsigned int[WIDTH*HEIGHT];
	for(int k=0;k<(WIDTH*HEIGHT);++k)
		res_pixels[k] = RGB(h_pixels[k].r,h_pixels[k].g,h_pixels[k].b);

	SaveBitmap(res_pixels,"Image");

	delete[] res_pixels;
	delete[] h_pixels;
	delete[] h_seeds;
	delete[] h_p;
	cudaFree(d_seedsCount);
	cudaFree(d_seeds);
	cudaFree(d_pitch);
	cudaFree(d_pixels);
	cudaFree(d_pCount);
	cudaFree(d_p);

	system("Image.bmp");
	std::cout<<std::endl;
	system("pause");

	return 0;
}

void LoadBitmap(unsigned int** pixels,unsigned int& width, unsigned int& height, const std::string Name)
{
	std::ifstream Input((Name+".bmp").c_str(), std::ios::binary);
	if( !Input.is_open() )
		abort();

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
	if( bitsPerPixel != 24 )//nomas leo bitmap de 24 bits
		abort();
	
	Input.seekg(offset,std::ios::beg);

	BGR* colors=new BGR[width*height];
	for(int k=0;k <(width*height);++k)
		Input.read((char*)&(colors[k]),sizeof(BGR));

	*pixels=new unsigned int[width*height];
	for(int k=0;k <(width*height);++k)
		(*pixels)[k]= RGB(colors[k].b,colors[k].g,colors[k].r);//lo guardo como RGB
	
	delete[] colors;
	
	Input.close();
	
		//en bitmap se guardan invertidos
	std::reverse(*pixels,*pixels+width*height);
	for(int y=0;y<height;++y)
		std::reverse(*pixels+y*width,*pixels+(y+1)*width);

	return;
}

void SaveBitmap(const unsigned int const* pixels,const std::string Name)
{
		//hago una copia para no modificar el parametro, guardandolo formateado
	RGBQUAD* my_pixels=new RGBQUAD[WIDTH*HEIGHT];
	for(int k=0;k<(WIDTH*HEIGHT);++k)
	{
		my_pixels[k].rgbRed      = pixels[k];
		my_pixels[k].rgbGreen    = pixels[k]>>8;
		my_pixels[k].rgbBlue     = pixels[k]>>16;
		my_pixels[k].rgbReserved = 0;
	}

		//lo invirto de forma que quede en formato bitmap
	std::reverse(my_pixels,my_pixels+WIDTH*HEIGHT);
	for(int i=0;i < HEIGHT;++i)
	{
		std::reverse(my_pixels+i*WIDTH,my_pixels+i*WIDTH+WIDTH);
	}

	BITMAPFILEHEADER BMHeader;
	BMHeader.bfType= 0x4D42;//MB -> swap -> BM			77 d-> 4D hex	66 d -> 42 hex
	BMHeader.bfReserved1=0;
	BMHeader.bfReserved2=0;
	BMHeader.bfSize= sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+WIDTH*HEIGHT*sizeof(RGBQUAD);
	BMHeader.bfOffBits= sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER);
	
	//BITMAPINFO BMInfo;	Esto se compone por BITMAPINFOHEADER + Pixels

	BITMAPINFOHEADER BMInfoHeader;
	ZeroMemory(&BMInfoHeader,sizeof(BMInfoHeader));
	BMInfoHeader.biBitCount= 32;
	BMInfoHeader.biPlanes=1;
	BMInfoHeader.biSize = sizeof(BMInfoHeader);
	BMInfoHeader.biHeight = HEIGHT;
	BMInfoHeader.biWidth = WIDTH;
	BMInfoHeader.biCompression=BI_RGB;
	BMInfoHeader.biSizeImage= 0;

	HANDLE Image =  CreateFile((Name+".bmp").c_str(),GENERIC_WRITE,NULL,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);

	if( !Image )
		exit(1);

	int byteW=0;
	WriteFile(Image,(const void*)&BMHeader,sizeof(BITMAPFILEHEADER),(LPDWORD)&byteW,NULL);

	WriteFile(Image,(const void*)&BMInfoHeader,sizeof(BITMAPINFOHEADER),(LPDWORD)&byteW,NULL);

	WriteFile(Image,(const void*)my_pixels,sizeof(RGBQUAD)*WIDTH*HEIGHT,(LPDWORD)&byteW,NULL);

	CloseHandle(Image);
	delete[] my_pixels;

	return;
}



