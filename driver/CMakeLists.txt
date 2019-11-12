set(TENSOR_SOURCE 
    src/tensor.cpp;
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
    set(DRIVER_SOURCE src/conv_driver.cpp)
elseif(DEVICE_BACKEND STREQUAL "NVIDIA")
    set(DRIVER_SOURCE src/conv_driver.cu)
endif()

add_executable(conv ${DRIVER_SOURCE}) 
target_link_libraries(conv PRIVATE host)
