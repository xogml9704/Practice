#include <stdio.h>

void main() {
    char A[] = "ARRAY";
    char *p = "POINTER";
    int i;
    for(i = 0; i<5; i++) 
    printf("*(A+%d) : %c\n", i, *(A+i)); // 배열을 포인터 형식으로 참조
    for(i=0; i<7; i++)
    printf("p[%d] : %c\n", i, p[i]); // 포인터를 배열 형식으로 참조
}