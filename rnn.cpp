#include "Halide.h"
#include "iostream"

//TODO Buffer/ Func implements of mat_vec and mat_mul
template<typename T>
void mat_mul(Halide::Buffer<T>& A, Halide::Buffer<T>& B, Halide::Func& forward
							, std::string mul_index, int num_threads = 4){

    int mul_rows = A.height();
    int mul_cols = B.width();

    Halide::Var x, y, z;
    Halide::RDom r(0, A.width());
    Halide::Func prod;

    prod(x, y, z) = A(x, z) * B(z, y);
    forward(x, y) = Halide::sum(x, y, r.x);

    //TODO optimized schedule for matrix multiplication
}
template <typename T>
void mat_vec_mul(Halide::Buffer<T>& mat, Halide::Buffer<T>& vec, Halide::Func& forward
							, std::string mul_index, int num_threads = 4){
	
	Halide::Var x("x at " + mul_index), y("y at " + mul_index), res_x("x at result " + mul_index);
	Halide::Var resx_i, resx_o, y_i, y_o;
	Halide::Func prod("prod at " + mul_index);
	Halide::RDom r(0, vec.width());

	//Algorithm
	prod(x, y) = mat(x, y) * vec(x);
	forward(res_x) = Halide::sum(prod(r.x, res_x));

	//Schedule -> vectorize columns, parallelize rows
	forward.split(res_x, resx_o, resx_i, ((width/ num_threads < 1) ? 1 : width/num_threads)
				,Halide::TailStrategy::GuardWithIf);
	forward.parallel(resx_o);
	prod.vectorize(x, 4).compute_at(forward, resx_o); //compute prod in chunks of thread

}

void mat_vec_mul(Halide::Func& mat, Halide::Func& vec, Halide::Func& forward
							, int width, std::string mul_index, int num_threads = 4){
	
	Halide::Var x("x at " + mul_index), y("y at " + mul_index), res_x("x at result " + mul_index);
	Halide::Var resx_i, resx_o, y_i, y_o;
	Halide::Func prod("prod at " + mul_index);
	Halide::RDom r(0, width);

	//Algorithm
	prod(x, y) = mat(x, y) * vec(x);
	forward(res_x) = Halide::sum(prod(r.x, res_x));

	//Schedule -> vectorize columns, parallelize rows
	forward.split(res_x, resx_o, resx_i, ((width/ num_threads < 1) ? 1 : width/num_threads)
				,Halide::TailStrategy::GuardWithIf);
	forward.parallel(resx_o);
	prod.vectorize(x, 4).compute_at(forward, resx_o); //compute prod in chunks of thread

}

Halide::Func softmax(){
    //TODO Softmax in Halide
    Halide::Func forward;
    return forward;
}

enum class Store{ NONE = 0, STATE = 1, OUTPUT = 2, BOTH = 3};

template <typename T>
class RNN{
  public:
    RNN(int word_dim, int hidden_dim, std::string layerName)
        : word_vec_dim(word_dim)
         ,hidden_unit_dim(hidden_dim)
         ,layerName(layerName)
         ,U(Halide::Buffer(word_dim, hidden_dim, "U at " + layerName))
         ,V(Halide::Buffer(hidden_dim, word_dim, "V at " + layerName))
         ,W(Halide::Buffer(hidden_dim, hidden_dim, "W at " + layerName))
         ,x("x at " + layerName)
         ,y("y at " + layerName){

        U(x, y) = Halide::cast<T>(Halide::random_float());
        V(x, y) = Halide::cast<T>(Halide::random_float());
        W(x, y) = Halide::cast<T>(Halide::random_float());
    }
    RNN(int word_dim, int hidden_dim, std::vector<Halide::Buffer<T>>& params)
        : word_vec_dim(word_dim)
         ,hidden_unit_dim(hidden_dim){
        
        assert(params.size() == 3 && "Size of param vector should be 3");
        U = Halide::Buffer(params[0]);
        V = Halide::Buffer(params[1]);
        W = Halide::Buffer(params[2]);
    }
    void feed_forward(std::vector<Halide::Buffer<T>>& seq, Store st){ 
        int total_steps = seq.size();
        for(int t = 0; t < total_steps; t++){
            Halide::Func calculate_state, calculate_output, mul_1, mul_2, mul_3;
            Halide::Var state_x, state_y, output_x, output_y;
            Halide::Buffer<T> state(hidden_unit_dim, hidden_unit_dim, "state " + " at " + layerName);
            mat_mul(U, seq[t], mul_1, "mul_1");
            if(t == 0){
                calculate_state(state_x, state_y) = Halide::tanh(mul_1(state_x, state_y));
                state.realize(calculate_state);
                states.push_back(state);
                if(st == Store::OUTPUT || st == Store::BOTH || total_steps == 0){
                    mat_mul(V, state, mul_3, "mul_3");
                    calculate_output(output_x, output_y) = softmax(mul_3(output_x, output_y));
                    Halide::Buffer<T> output(word_dim, "output");
                    outputs.realize(calculate_output);
                    outputs.push_back(output);
                }
                continue;
            }
            if(st == Store::STATE || Store::BOTH)
                mat_mul(W, states[t - 1], mul_2, "mul_2");
            else
                mat_mul(W, states[0], mul_2, "mul_2");
            
            calculate_state(state_x, state_y) = 
                Halide::tanh(mul_1(state_x, state_y) + mul_2(state_x, state_y));
            state.realize(calculate_state);
            
            if(st == Store::STATE || Store::BOTH)
                states.push_back(state);
            else
                states[0] = state;
            if(st == Store::OUTPUT || st == Store::BOTH || t = (total_steps - 1)){
                mat_mul(V, state, mul_3, "mul_3");
                calculate_output(output_x, output_y) = softmax(mul_3(output_x, output_y));
                Halide::Buffer<T> output(word_dim, "output");
                outputs.realize(calculate_output);
                outputs.push_back(output);
            }
        }
    }
    
    void back_propagate(){
        //TODO back Prop in Halide
    }
  private:
    int word_vec_dim;
    int hidden_unit_dim;
    std::string layerName;

    std::vector<Halide::Buffer<T>> states;
    std::vector<Halide::Buffer<T>> outputs;
    Halide::Buffer<T> U;
    Halide::Buffer<T> V;
    Halide::Buffer<T> W;

    Halide::Var x, y;
};

int main(){
	Halide::Buffer<uint32_t> mat(16, 32, "matrix");
	Halide::Buffer<uint32_t> vec(16, "vector");
	Halide::Buffer<uint32_t> out(32, "output");

	//initialize
	for(int i = 0; i < 16; i++){
		for(int j = 0; j < 32; j++)
			mat(i, j) = 1;
		vec(i) = 2;
	}

	
	Halide::Func res("result"); 
	mat_vec_mul<uint32_t>(mat, vec, res, "1");
	res.realize(out);

	for(int i = 0; i < 32; i++)
		std::cout << out(i) << std::endl;
	return 0;
}
