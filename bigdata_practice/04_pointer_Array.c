#include <stdio.h>
void main() {
    int days = 365;
    int month = 12;
    int Table[5] = {1,2,3,4,5};
    printf("days의 주소는 %x\n", &days);
    printf("month의 주소는 %x\n", &month);
    printf("배열명 Table의 주소는 %x\n", Table);
    // 배열명은 주소를 나타냄 //
    printf("배열명 Table 첫번째 요소의 주소는 %x\n", &Table[0]);
    printf("배열명 Table 두번쨰 요소의 주소는 %x\n", &Table[1]);
}