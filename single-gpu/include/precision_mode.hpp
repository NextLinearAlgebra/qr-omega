#pragma once
// Precision modes. fp32 storage runs in one of three math modes:
//  ieee : IEEE single products (TF32 disabled) -- the default;
//  tf32 : W/Z/D products on TF32 tensor cores (m16n8k8), validated at TF32 unit roundoff 2^-11;
//  x3   : error-compensated 3xTF32 (a = big + small, big*big + big*small + small*big, fp32
//         accumulation; Ootomo-Yokota style) validated at the unchanged fp32 tolerance.
// Selected by --fp32-math (family binary) or the TQR_FP32_MATH environment variable (every binary);
// recorded in every result and witness. fp64 ignores it.
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace tqr {
enum class Fp32Math{IEEE=0,TF32=1,X3=2};
inline Fp32Math parse_fp32_math(const std::string&s){
 if(s=="ieee"||s.empty())return Fp32Math::IEEE;if(s=="tf32")return Fp32Math::TF32;if(s=="x3"||s=="3xtf32")return Fp32Math::X3;
 throw std::runtime_error("unknown_fp32_math:"+s);}
inline Fp32Math& fp32_math(){
 static Fp32Math m=[]{const char*e=std::getenv("TQR_FP32_MATH");return parse_fp32_math(e?e:"");}();return m;}
inline const char* fp32_math_name(Fp32Math m){return m==Fp32Math::TF32?"tf32":m==Fp32Math::X3?"x3":"ieee";}
// The unit roundoff the validation tolerances are derived from.
template<class T> double working_unit_roundoff(){
 if constexpr(std::is_same_v<T,float>){if(fp32_math()==Fp32Math::TF32)return std::ldexp(1.0,-11);}
 return std::numeric_limits<T>::epsilon()/2;}
template<class T> const char* precision_mode_name(){
 if constexpr(std::is_same_v<T,float>)return fp32_math()==Fp32Math::TF32?"tf32":fp32_math()==Fp32Math::X3?"fp32x3":"fp32";
 else return "fp64";}
}
