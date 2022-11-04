# 检索src目录查找cpp为后缀的文件，用shell指令的find
srcs := $(shell find src -name "*.cpp")

cu_srcs := $(shell find src -name "*.cu")

# 将srcs的后缀为.cpp的替换为.o
objs := $(srcs:.cpp=.o)

# 将src/前缀替换为objs前缀，让.o文件方到objs目录下
objs := $(objs:src/%=objs/%)

cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=objs/%)


# 1. 增加include_paths选项, 因为头文件需要他们
include_paths := ./include \
				 /usr/local/cuda-10.2/targets/x86_64-linux/include
opencv_include := `pkg-config opencv --cflags`
onnxruntime_include := /usr/local/include/onnxruntime/core/session \
					   /usr/local/include/onnxruntime/core/providers/cuda

# 2. 增加ld_liberarys选项，因为链接需要他们
library_paths := /usr/local/lib \
				 /usr/local/cuda-10.2/targets/x86_64-linux/lib
ld_librarys := m pthread
onnx_librarys := onnxruntime
cuda_librarys :=  cuda curand cublas cudart
ld_librarys += $(onnx_librarys)
ld_librarys += $(cuda_librarys)
opencv_librarys := `pkg-config opencv --libs`

# 3. 将每一个头肩路径前面增加-I, 库文件路径前面增加-L, 链接选项前面加上-l
# -I 配置头文件路径
# -L 配置库路径
# -lname 配置依赖的so
# 增加run path 变量，语法为
# g++ main.o test.o -o out.bin -Wl,-rpath=/path1/lib  /path2/lib
run_paths := $(library_paths:%=-Wl,--rpath=%)
include_paths += $(onnxruntime_include)
include_paths := $(include_paths:%=-I%)
include_paths += $(opencv_include)
library_paths := $(library_paths:%=-L%)
ld_librarys   := $(ld_librarys:%=-l%)
ld_librarys += $(opencv_librarys)

# 4. 增加compile_flags,增加编译选项,例如我们需要C++11特性等,
# -w避免警告,-g生成调试信息
# -O0优化级别关闭
compile_flags := -std=c++11 -w -g -O0 $(include_paths)
cu_compile_flags := -std=c++11 -Xcompiler -fPIC -w -g -O0 $(include_paths)
link_flags := $(library_paths) $(ld_librarys) $(run_paths)

# 定义objs下的o文件，依赖src下对应的cpp文件
# $@ = 左边的生成项
# $< = 右边的依赖项第一个
# 5. 将编译选项增加到g++编译后面
objs/%.o : src/%.cpp
	@echo "\033[32m Build CXX object $@ \033[0m"
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(compile_flags)

objs/%.cuo : src/%.cu
	@echo "\033[32m Build CUDA object $@ \033[0m"
	@mkdir -p $(dir $@)
	@nvcc -c $< -o $@ $(cu_compile_flags)

# $^ = 右边的依赖项全部
# 6. 将链接选项增加到g++链接后面
workspace/pro : $(cu_objs) $(objs)
	@echo "\033[32m Linking CXX executable ,$@ \033[0m"
	@g++ $^ -o $@ $(link_flags)

# 定义简洁指令, make pro即可生成程序
pro : workspace/pro
	@echo Compile sucess

# 定义make run, 编译好pro后执行
run : pro
	@cd workspace && ./pro

# 定义make clean, 清理编译留下的垃圾
clean:
	@rm -rf workspace/pro objs

debug:
	@echo objs is [$(objs)]
	@echo $(ld_librarys)
	@echo $(run_paths)

.PHONY : pro run debug clean