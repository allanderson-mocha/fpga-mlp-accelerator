#include "system.h"
#include "altera_avalon_pio_regs.h"
#include <unistd.h>

int main() {
    while (1) {
        unsigned int sw = IORD_ALTERA_AVALON_PIO_DATA(SWITCHES_BASE);
        IOWR_ALTERA_AVALON_PIO_DATA(LEDS_BASE, sw);
        usleep(10000); // debounce delay
    }
}




