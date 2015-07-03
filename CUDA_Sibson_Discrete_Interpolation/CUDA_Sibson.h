#define WINDOWS
//#define MY_DEBUG

#ifdef WINDOWS
	#include <windows.h>
	#include <iostream>
	#include <fstream>
	#include <algorithm>
	#include <string>
#endif

//#include <cutil_inline.h>
//#include <cuda.h>

//Macros
#define WIDTH  96//1280
#define HEIGHT 96//720

#define FAR 0xFFFFFFFFFFFFFFFF;

//Host functions
void LoadBitmap(unsigned int**,unsigned int&,unsigned int&,const std::string);
void SaveBitmap(const unsigned int const*,const std::string);

//Structs

struct Seed
{
unsigned int x;
unsigned int y;

unsigned char r;
unsigned char g;
unsigned char b;
};

struct Pixel
{
unsigned char r;
unsigned char g;
unsigned char b;

//semilla mas cercana
Seed* SeedClosest;
//distancia a la semilla ams cercana
unsigned int SeedDst;
};

//Punto a interpolar
struct P
{
unsigned int x;
unsigned int y;

//se usa unsigned int y no unsigned char porque solo hay atomic function para enteros
//y al parecer no representa de la misma forma char e int el GPU, porque en la placa anda 
//mal pero en emulacion hace el calculo esperado
unsigned int r;
unsigned int g;
unsigned int b;
unsigned int n;
};

struct BGR//en bitmap se guarda bgr
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
};