#include "blake3.h"
extern "C" void pre_allocate();
extern "C" void post_free();
int main(){
    pre_allocate();
    post_free();
}
