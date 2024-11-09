from mochi_preview.t2v_synth_mochi import T2VSynthMochiModel


def noexcept(f):
    try:
        return f()
    except:
        pass


class MochiWrapper:
    def __init__(self, *, num_workers, **actor_kwargs):
        super().__init__()
        self.worker = T2VSynthMochiModel(device_id=0, world_size=num_workers, local_rank=0, **actor_kwargs)
        # RemoteClass = ray.remote(T2VSynthMochiModel)
        # self.workers = [
        #     RemoteClass.options(num_gpus=1).remote(
        #         device_id=0, world_size=num_workers, local_rank=i, **actor_kwargs
        #     )
        #     for i in range(num_workers)
        # ]
        # # Ensure the __init__ method has finished on all workers
        # for worker in self.workers:
        #     ray.get(worker.__ray_ready__.remote())
        self.is_loaded = True

    def __call__(self, args):
        # work_refs = [
        #     worker.run.remote(args, i == 0) for i, worker in enumerate(self.workers)
        # ]

        work_ref = self.worker.run(args, True)
        for result in work_ref:
            yield result
        # try:

        #     # # Handle the (very unlikely) edge-case where a worker that's not the 1st one
        #     # # fails (don't want an uncaught error)
        #     # for result in work_refs[1:]:
        #     #     ray.get(result)
        # except Exception as e:
        #     # # Get exception from other workers
        #     # for ref in work_refs[1:]:
        #     #     noexcept(lambda: ray.get(ref))
        #     raise e
