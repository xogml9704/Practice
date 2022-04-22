#include <stdio.h>

void main() {
    static int a[] = {10,20,30,40,50,};
    int *pt, b, c, d;
    pt = a; // 배열명을 사용하여 배열의 시작주소를 할당
    b = *pt + *(pt+3); // b = 10 + 40
    pt ++; // 포인터를 1이동
    c = *pt + *(pt+3); // c = 20 + 50
    d = *pt + 3; // d = 20 + 3
    printf("b=%d, c=%d, d=%d\n", b, c, d);
}