#include <gint_device.cuh>


int main()
{
    geval<<<1, 1>>>(nullptr, nullptr);
    return 0;
}
