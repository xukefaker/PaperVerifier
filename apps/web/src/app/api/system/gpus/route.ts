import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { NextResponse } from 'next/server';

const execFileAsync = promisify(execFile);

function splitCsv(line: string) {
  return line.split(',').map((part) => part.trim());
}

export async function GET() {
  try {
    const [{ stdout: gpuStdout }, { stdout: appStdout }] = await Promise.all([
      execFileAsync('nvidia-smi', ['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']),
      execFileAsync('nvidia-smi', ['--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits']).catch(() => ({ stdout: '' })),
    ]);
    const processesByUuid = new Map<string, { pid: string; name: string; used_memory_mb: number }[]>();
    for (const line of appStdout.trim().split('\n').filter(Boolean)) {
      const [uuid, pid, name, used] = splitCsv(line);
      processesByUuid.set(uuid, [...(processesByUuid.get(uuid) ?? []), { pid, name, used_memory_mb: Number(used) || 0 }]);
    }
    const gpus = gpuStdout.trim().split('\n').filter(Boolean).map((line) => {
      const [index, uuid, name, total, used, utilization] = splitCsv(line);
      return {
        index: Number(index),
        uuid,
        name,
        memory_total_mb: Number(total) || 0,
        memory_used_mb: Number(used) || 0,
        utilization_gpu: Number(utilization) || 0,
        processes: processesByUuid.get(uuid) ?? [],
      };
    });
    return NextResponse.json({ available: gpus.length > 0, cuda_visible_devices: process.env.CUDA_VISIBLE_DEVICES ?? '', gpus });
  } catch {
    return NextResponse.json({ available: false, cuda_visible_devices: process.env.CUDA_VISIBLE_DEVICES ?? '', gpus: [] });
  }
}
