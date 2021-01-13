
#include <cmath>
#include "imgproc.h"

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;
using namespace std;



ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}

void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}

bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
   org_img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->read_image(TypeDesc::FLOAT, org_img_data);
   in->close ();
   cout << "Nc: " << Nc << endl;
   return true;
}

void ImgProc::write(){
  const char* file_name = "output.exr";
  auto out = ImageOutput::create(file_name);
  if(!out){
    return;
  }
    ImageSpec spe(Nx, Ny, Nc, TypeDesc::FLOAT);
    out->open (file_name, spe);
    out->write_image(TypeDesc::FLOAT, img_data);
    out->close();
  }

//one pixel value, include RGB...
void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{

   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}
//set one value include RGB...
void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{

   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}


ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}

ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}

void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}

void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}

void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}

/*******************************************************/
/*******************************************************/

void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }
}


void ImgProc::brightness(float brit) {
  if( img_data == nullptr ){ return; }
  #pragma omp parallel for
  for( long i=0;i<Nsize;i++ ){
    img_data[i] = img_data[i]*brit;
  }

}


void ImgProc::bias(float bis) {
  #pragma omp parallel for
    for( long i=0;i<Nsize;i++ ){
      img_data[i] = img_data[i] + bis;
    }
}

void ImgProc::gamma (float gm) {
  #pragma omp parallel for
    for( long i=0;i<Nsize;i++ ){
      img_data[i] = pow(img_data[i],gm);
    }
}



void ImgProc::flap() {
    #pragma omp parallel for
  for(int j=0;j<Ny/2;j++)
  {
  #pragma omp parallel for
   for(int i=0;i<Nx;i++)
   {
      vector<float> C;
      vector<float> C2;
      value(i,j,C);
      value(i,Ny-j-1,C2);
      set_value(i,Ny-j-1,C);
      set_value(i,j,C2);
   }
  }

}

void ImgProc::grayscale() {


   //only need 2 pragma //always 3 channels
   float g = 0.0;
     #pragma omp parallel for
     for(long j = 0; j < Ny; j++ ) {
        #pragma omp parallel for
        for(long i = 0; i < Nx; i++) {
        vector<float> p_c(Nc);
        value(i,j,p_c);
        g = p_c[0]*0.2126 + p_c[1]*0.7152 + p_c[2]*0.0722;

        for(unsigned int c = 0; c < p_c.size(); c++){
           p_c[c] = g;
         }
         set_value(i,j,p_c);
       }
     }
   }


//calculate each channel 0<data<1 every R, G, and B
//result = floor(data*at[i][j])/data*at[i][j]
void ImgProc::quantize () {

   #pragma omp parallel for
   for(long i = 0; i < Ny; i++ ) {
     #pragma omp parallel for
     for(long j = 0; j < Nx; j++) {
       vector<float> p_c(Nc);
       value(j,i,p_c);
     //  #pragma omp parallel for
       for(long c = 0; c < Nc; c++) {
         int tmp = p_c[c]*(i*Ny + j);
          p_c[c] = tmp/(float)(i*Ny + j);
       }
       set_value(j,i,p_c);
     }
   }
}



void ImgProc::rms() {
    //step 1   computing mean
    float sum_r = 0.0;
    float sum_g = 0.0;
    float sum_b = 0.0;
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
            vector<float> p_c(Nc);
            value (j, i, p_c);

                sum_r += p_c[0];
                sum_g += p_c[1];
                sum_b += p_c[2];
        }
    }
    float mean_r = sum_r/(Nx*Ny);
    float mean_g = sum_g/(Nx*Ny);
    float mean_b = sum_b/(Nx*Ny);
    //step2 computing sigma
    float vari_r = 0;
    float vari_g = 0;
    float vari_b = 0;
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
            vector<float> p_c(Nc);
            value (j, i, p_c);
            vari_r += pow((p_c[0] - mean_r),2);
            vari_g += pow((p_c[1] - mean_g),2);
            vari_b += pow((p_c[2] - mean_b),2);
        }
    }

    float sigma_r = sqrt(vari_r/(Nx*Ny));
    float sigma_g = sqrt(vari_g/(Nx*Ny));
    float sigma_b = sqrt(vari_b/(Nx*Ny));

    //step 3  computing contrast and set value to data
    #pragma omp parallel for
    for(long i = 0; i < Ny; i++) {
        #pragma omp parallel for
        for(long j = 0; j < Nx; j++) {
          vector<float> p_c(Nc);
          value(j,i,p_c);
          p_c[0] = (p_c[0] - mean_r)/sigma_r;
          p_c[1] = (p_c[1] - mean_g)/sigma_g;
          p_c[2] = (p_c[2] - mean_b)/sigma_b;
          set_value(j,i,p_c);

       }
    }
}



void ImgProc::org_data() {
  for(long i = 0; i<Nsize; i++) {
    img_data[i] = org_img_data[i];
  }
}


/**************************************************/
/*********************************************/

long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}

void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}
