#include <stdio.h>
void main() {
    char a ='A', *p, **pp;
    p = &a;
    pp = &p;
    printf("**pp = %c", **pp);
}