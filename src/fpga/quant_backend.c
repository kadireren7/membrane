#include "membrane/quant_backend.h"

membrane_quant_backend_t	membrane_choose_quant_backend(
			membrane_quant_backend_t requested,
			uint32_t batch_blocks, uint32_t fpga_queue_used,
			uint32_t fpga_queue_depth, uint32_t cpu_cores_available)
{
	if (requested != MEMBRANE_QUANT_BACKEND_AUTO)
		return (requested);
	if (fpga_queue_depth == 0)
		return (MEMBRANE_QUANT_BACKEND_CPU);
	if (fpga_queue_used >= fpga_queue_depth)
		return (MEMBRANE_QUANT_BACKEND_CPU);
	if (cpu_cores_available >= 4)
		return (MEMBRANE_QUANT_BACKEND_CPU);
	if (batch_blocks >= MEMBRANE_QUANT_FPGA_BREAK_EVEN_BLOCKS)
		return (MEMBRANE_QUANT_BACKEND_FPGA);
	return (MEMBRANE_QUANT_BACKEND_CPU);
}
