set(TENSOR_SOURCE 
    src/host_tensor.cpp;
    src/device.cpp;
)

add_library(host SHARED ${TENSOR_SOURCE})
target_compile_features(host PUBLIC)
set_target_properties(host PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DEVICE_BACKEND STREQUAL "NVIDIA")
    target_link_libraries(host nvToolsExt cudart)
endif()

install(TARGETS host LIBRARY DESTINATION lib) 


if(DEVICE_BACKEND STREQUAL "AMD")
    set(CONV_SOURCE src/conv_driver.cpp)
    set(CONV_BWD_DATA_SOURCE src/conv_bwd_data_driver.cpp)
    set(TRY_SOURCE src/col2im_driver.cpp)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(CONV_SOURCE src/conv_driver.cu)
    set(CONV_BWD_DATA_SOURCE src/conv_bwd_data_driver.cu)
    set(TRY_SOURCE src/col2im_driver.cu)
endif()

add_executable(conv_driver ${CONV_SOURCE}) 
add_executable(conv_bwd_data_driver ${CONV_BWD_DATA_SOURCE}) 
add_executable(col2im_driver ${TRY_SOURCE}) 

target_link_libraries(conv_driver PRIVATE host)
target_link_libraries(conv_bwd_data_driver PRIVATE host)
target_link_libraries(col2im_driver PRIVATE host)
