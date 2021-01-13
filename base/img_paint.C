//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------




#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>



#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;

ImgProc image;


//float


void setNbCores( int nb )
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
}

void cbMouse( int button, int state, int x, int y )
{
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGB, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();
}

void cbOnKeyboard( unsigned char key, int x, int y )
{
   switch (key)
   {
      case 'c':
    	 image.compliment();
    	 cout << "Compliment\n";
    	 break;
      case 'V':
       //brit *=1.1;
        image.brightness(1.05);
        cout << "brightness increase\n";
        break;
      case 'v':
      //  brit *= 0.90;
        image.brightness(0.95);
        cout << "brightness decrease\n";
        break;
      case 'B':

        image.bias(0.05);
        cout << "bias increase\n";
        break;
      case 'b':

        image.bias(-0.05);
        cout << "bias decrease\n";
        break;

      case 'f' :
        image.flap();
        cout << "flap\n";
        break;

      case 'G':
       // gm*=1.1;
        image.gamma(1.1);

         cout << "increase gamma\n";
         break;
      case 'g':

        image.gamma(0.95);
        cout << "decrease gamma\n";
        break;
      case 'q':
        image.quantize();
        cout << "quantize\n";
        break;
      case 'w':
        image.grayscale();
        cout << "grayscale\n";
        break;
     case 'C' :
        image.rms();
        cout << "rms contrast\n";
        break;
      case 'r':
        image.org_data();
          cout <<"original data\n" << endl;
      case 'o' :
        image.write();
        cout << "write file\n";

   }
}

void PrintUsage()
{
   cout << "img_paint keyboard choices\n";
   cout << "c         compliment\n";
   cout << "V         brightness increase\n";
   cout << "v         brightness decrease\n";
   cout << "B         bias increase\n";
   cout << "b         bias decrease\n";
   cout << "f         flap\n";
   cout << "G         increase gamma\n";
   cout << "g         decrease gamma\n";
   cout << "q         quantize\n";
   cout << "w         grayscale\n";
   cout << "R         original data\n";
   cout << "o         write out_file \n" << endl;

}


int main(int argc, char** argv)
{
   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   string imagename = clf.find("-image", "", "Image to drive color");

   clf.usage("-h");
   clf.printFinds();
   PrintUsage();

   image.load(imagename);


   // GLUT routines
   glutInit(&argc, argv);

   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize( image.nx(), image.ny() );

   // Open a window
   char title[] = "img_paint";
   glutCreateWindow( title );

   glClearColor( 1,1,1,1 );

   glutDisplayFunc(&cbDisplay);
   glutIdleFunc(&cbIdle);
   glutKeyboardFunc(&cbOnKeyboard);
   glutMouseFunc( &cbMouse );
   glutMotionFunc( &cbMotion );

   glutMainLoop();
   return 1;
};
