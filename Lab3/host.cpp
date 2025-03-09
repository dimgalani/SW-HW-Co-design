#include "xcl2.hpp"
#include "event_timer.hpp"
#include <algorithm>
#include <vector>
#include <numeric>
#include <ap_int.h>

#define L_M 4
#define L_N 4
#define L_P 4
#define M (1 << L_M)
#define N (1 << L_N)
#define P (1 << L_P)

#define DATA_SIZE 256 //check bigger

#define ELEMENTS 16
//#define DATA_SIZE 16384

typedef ap_uint<512> uint512_dt;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  EventTimer et;

  std::string binaryFile = argv[1];
  size_t vector_size_bytes = sizeof(uint512_dt) * ELEMENTS;
  size_t output_vector_size_bytes = sizeof(uint512_dt) * ELEMENTS*2;
//  size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
// size_t output_vector_size_bytes = sizeof(int) * DATA_SIZE *2;
  cl_int err;
  cl::Context context;
  cl::Kernel krnl_vector_add;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR
  et.add("Allocate Memory in Host Memory");
  // std::vector<int, aligned_allocator<int>> source_in1(DATA_SIZE);
  // std::vector<int, aligned_allocator<int>> source_in2(DATA_SIZE);
  // std::vector<int64_t, aligned_allocator<int64_t>> source_hw_results(DATA_SIZE);
  // std::vector<int64_t, aligned_allocator<int64_t>> source_sw_results(DATA_SIZE);

  // Create arrays of 512-bit integers
  std::vector<uint512_dt, aligned_allocator<uint512_dt>> source_in1(ELEMENTS);
  std::vector<uint512_dt, aligned_allocator<uint512_dt>> source_in2(ELEMENTS);
  std::vector<uint512_dt, aligned_allocator<uint512_dt>> transposed(ELEMENTS);
  std::vector<uint512_dt, aligned_allocator<uint512_dt>> device_output(2*ELEMENTS);
  std::vector<uint512_dt, aligned_allocator<uint512_dt>> sw_results(2*ELEMENTS);

//  std::vector<unsigned int, aligned_allocator<unsigned int>> source_in1(DATA_SIZE);
//  std::vector<unsigned int, aligned_allocator<unsigned int>> source_in2(DATA_SIZE);
//  std::vector<unsigned int, aligned_allocator<unsigned int>> transposed(DATA_SIZE);
//  std::vector<uint64_t, aligned_allocator<uint64_t>> device_output(DATA_SIZE);
//  std::vector<uint64_t, aligned_allocator<uint64_t>> sw_results(DATA_SIZE);

  et.finish();

  // Create the test data
  et.add("Fill the buffers");

  for (int i = 0; i < ELEMENTS; i++) {
      uint512_dt temp1 = 0, temp2 = 0; // Temporary 512-bit integers
      for (int j = 0; j < 16; j++) {
          // Set each 32-bit slot within the 512-bit element
          temp1.range(32 * (j + 1) - 1, 32 * j) = std::rand();
          temp2.range(32 * (j + 1) - 1, 32 * j) = std::rand();
//          temp1.range(32 * (j + 1) - 1, 32 * j) = (std::rand() % 1073741823) + 1;
//          temp2.range(32 * (j + 1) - 1, 32 * j) = (std::rand() % 1073741823) + 1;
      }
      source_in1[i] = temp1; // Assign to source_in1
      source_in2[i] = temp2; // Assign to source_in2
  }

//  std::generate(source_in1.begin(), source_in1.end(), std::rand);
//  std::generate(source_in2.begin(), source_in2.end(), std::rand);

    // Transpose the source_in2 array
  for (int i = 0; i < 16; i++) { // Loop over rows
      uint512_dt row = 0;
      for (int j = 0; j < 16; j++) { // Loop over columns
          // Extract the value at (j, i) and set it at (i, j)
          int value = source_in2[j].range(32 * (i + 1) - 1, 32 * i).to_int();
          row.range(32 * (j + 1) - 1, 32 * j) = value;
      }
      transposed[i] = row; // Assign transposed row to the result
  }


  // Print the transposed data for verification
  std::cout << "Transposed source_in2:" << std::endl;
  for (const auto& elem : transposed) {
      for (int j = 0; j < 16; j++) {
          std::cout << elem.range(32 * (j + 1) - 1, 32 * j).to_int() << " ";
      }
      std::cout << std::endl;
  }

  // Print the generated data for verification
  std::cout << "source_in1:" << std::endl;
  for (const auto& elem : source_in1) {
      for (int j = 0; j < 16; j++) {
          std::cout << elem.range(32 * (j + 1) - 1, 32 * j).to_int() << " ";
      }
      std::cout << std::endl;
  }

  std::cout << "source_in2:" << std::endl;
  for (const auto& elem : source_in2) {
      for (int j = 0; j < 16; j++) {
          std::cout << elem.range(32 * (j + 1) - 1, 32 * j).to_int() << " ";
      }
      std::cout << std::endl;
  }

    // Matrix multiplication
  for (int i = 0; i < 16; i++) { // Rows of source_in1
      for (int j = 0; j < 16; j++) { // Columns of transposed source_in2
          int64_t sum = 0;
          for (int k = 0; k < 16; ++k) { // Dot product calculation
              int value1 = source_in1[i].range(32 * (k + 1) - 1, 32 * k).to_int();
              int value2 = source_in2[k].range(32 * (j + 1) - 1, 32 * j).to_int();
              sum += static_cast<int64_t>(value1) * static_cast<int64_t>(value2);
          }
          int flat_index = i*16+j;
          int chunk_index = flat_index / 8;
          int offset = flat_index % 8;
          sw_results[chunk_index].range(64 * (offset + 1) - 1, 64 * offset) = sum; // Store results
      }
  }

  // Print the resulting sw_results for verification
  std::cout << "Matrix Multiplication Results (sw_results):" << std::endl;
  for (const auto& elem : sw_results) {
      for (int j = 0; j < 8; j++) {
          std::cout << elem.range(64 * (j + 1) - 1, 64 * j).to_int64() << " ";
      }
      std::cout << std::endl;
  }

  et.finish();

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  et.add("Load Binary File to Alveo U200");
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vadd", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }
  et.finish();

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication

  et.add("Allocate Buffer in Global Memory");


  OCL_CHECK(err, cl::Buffer buffer_in1(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in1.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_in2(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, transposed.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
					 output_vector_size_bytes, device_output.data(), &err));


//  OCL_CHECK(err, cl::Buffer buffer_in1(
//                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
//                     vector_size_bytes, source_in1.data(), &err));
//  OCL_CHECK(err, cl::Buffer buffer_in2(
//                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
//                     vector_size_bytes, transposed.data(), &err));
//  OCL_CHECK(err, cl::Buffer buffer_output(
//                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
//					 output_vector_size_bytes, device_output.data(), &err));

  et.finish();

  et.add("Set the Kernel Arguments");
  int size = DATA_SIZE;
  OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in2));
  OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_output));
  OCL_CHECK(err, err = krnl_vector_add.setArg(3, size));
  et.finish();

  // Copy input data to device global memory
  et.add("Copy input data to device global memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));
  et.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel
  et.add("Launch the Kernel");
  OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
  et.finish();

  // Copy Result from Device Global Memory to Host Local Memory
  et.add("Copy Result from Device Global Memory to Host Local Memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
  OCL_CHECK(err, err = q.finish());
  et.finish();
  // OPENCL HOST CODE AREA END

  // Compare the results of the Device to the simulation
  et.add("Compare the results of the Device to the simulation");

  // Print both arrays for debugging (optional)
  std::cout << "Software Results (sw_results):" << std::endl;
  for (const auto& val : sw_results) {
      for (int i = 0; i < 8; i++) {
          std::cout << val.range(64 * (i + 1) - 1, 64 * i).to_uint64() << " ";
      }
      std::cout << std::endl;
  }

  std::cout << "Hardware Results (device_output):" << std::endl;
  for (const auto& val : device_output) {
      for (int i = 0; i < 8; i++) {
          std::cout << val.range(64 * (i + 1) - 1, 64 * i).to_uint64() << " ";
      }
      std::cout << std::endl;
  }

  // Compare the results
  bool match = true;
  for (size_t i = 0; i < sw_results.size(); i++) {
      for (int j = 0; j < 8; j++) {
          uint64_t sw_value = sw_results[i].range(64 * (j + 1) - 1, 64 * j).to_uint64();
          uint64_t hw_value = device_output[i].range(64 * (j + 1) - 1, 64 * j).to_uint64();
          if (sw_value != hw_value) {
              std::cout << "Error: Result mismatch" << std::endl;
              std::cout << "i = " << i << ", j = " << j
                        << " CPU result = " << sw_value
                        << " Device result = " << hw_value << std::endl;
              match = false;
          }
      }
  }

  et.finish();
  std::cout <<"----------------- Key execution times -----------------" << std::endl;
  et.print();

  // Final result
  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return match ? EXIT_SUCCESS : EXIT_FAILURE;
  }