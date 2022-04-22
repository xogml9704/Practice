#include <stdio.h>
void main() {
    char name[] = "HONG GIL DONG";
    char adrs1[6] = {'S', 'E', 'O', 'U', 'L', '\0'};
    // 마지막 요소를 \O으로 함
    char adrs2[6] = {'S', 'E', 'O', 'U', 'L'};
    // 마지막 요소를 \O으로 하지 않음
    printf("\n name : %s", name);
    printf("\n adrs1 : %s", adrs1);
    printf("\n adrs2 : %s\n", adrs2);
}