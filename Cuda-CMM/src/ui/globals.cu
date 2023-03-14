
#include "globals.h"

// GLOBAL ERROR HANDLING
// please use this function to handle errors
// it will print the error message and exit the program
// with the given error code, which is unique for each error
int error(std::string err_msg, int err_code)
{
    std::cout << err_msg + "  | " +  std::to_string(err_code) << std::endl;
    exit(err_code);
}
