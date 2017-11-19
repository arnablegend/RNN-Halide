#include "iostream"
#include "Halide.h"

int main(){
    Halide::Var x("x"), y("y"), z("z");
    Halide::Func preprocess, conv("Convolution 3d"), relu("RELU");
    Halide::Buffer<int32_t> ifm(16, 16, 3, "IFM"), kernel(3, 3, 3, 4);

    Halide::RDom r(0, 3, 0, 3, 0, 3);

    //Initialize
    for(int k =0; k < 4; k++)
        for(int j = 0; j < 3; j++)
            for(int i = 0; i < 3; i++)
                kernel(0, i, j, k) = kernel(1, i, j, k)
                  = kernel(2, i, j, k) = 1;

    for(int k = 0; k < 3; k++)
        for(int j = 0; j < 16; j++)
            for(int i = 0; i < 16; i++)
                ifm(i, j, k) = 2;

    //Algorithm
    preprocess = Halide::BoundaryConditions::constant_exterior(
                    ifm, 0, 0, 16, 0, 16, 0, 3);

    conv(x, y, z) = Halide::sum(preprocess(x + r.x - 1
        , y + r.y -1, r.z) * kernel(r.x, r.y, r.z, z));

    relu(x, y, z) = Halide::max(0, conv(x, y, z));

    //Schedule
    conv.reorder(x, y, z);
    relu.compute_at(conv, x);
    
    //conv.trace_stores().trace_loads();
    //relu.trace_stores();
    relu.print_loop_nest(); 
    Halide::Buffer<int32_t> out(16, 16, 4, "output");
    relu.realize(out);

    return 0;
}
