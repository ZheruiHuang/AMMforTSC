import psutil

def computer_monitor(max_mem_used=90, max_cpu_used=100):
    mem_used_percent = psutil.virtual_memory().percent
    cpu_used_percent = psutil.cpu_percent()
    # print('computer_monitor is running...')
    # print(f'memory_used_percent: {mem_used_percent}\ncpu_used_percent: {cpu_used_percent}')
    if mem_used_percent > max_mem_used or cpu_used_percent > max_cpu_used:
        print(f'memory_used_percent: {mem_used_percent}\ncpu_used_percent: {cpu_used_percent}')
        raise OSError('Calculate resource limit exceed.')
    
