#include <stdio.h>
void main() {
    int *p, i=3, j;
    p = &i; // 포인터 변수 p는 변수 i의 주소를 가리킴
    j = *p; // 포인터 변수 p가 가리키는 번지의 내용을 변수 j에 대입
    j++; // 포인터 변수 j 의 값을 1 증가
    printf("*p = %d\n", *p);
    printf(" p = %x\n", p);
    printf(" j = %d\n", j);
