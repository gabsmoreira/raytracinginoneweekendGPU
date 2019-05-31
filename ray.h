#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        _device_ ray() {}
        _device_ ray(const vec3& a, const vec3& b) { A = a; B = b; }  
        _device_ vec3 origin() const       { return A; }
        _device_ vec3 direction() const    { return B; }
        _device_ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif