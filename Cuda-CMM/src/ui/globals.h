/******************************************************************************************************************************
 * FILENAME:          globals.h
 *
 *  AUTHORS:           Philipp Krah
 * ******************************************************************************************************************************/

// header file for settings
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>


// GLOBAL ERROR HANDLING
// please use this function to handle errors
// it will print the error message and exit the program
// with the given error code, which is unique for each error
int error(std::string err_msg, int err_code);

#endif
