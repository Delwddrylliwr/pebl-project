"""Classes and functions for running tasks on Amazon's EC2"""

import time
import os.path
import shutil 
import tempfile
import sys

from pebl import config, result
from pebl.taskcontroller.ipy1 import IPy1Controller, IPy1DeferredResult


class EC2DeferredResult(IPy1DeferredResult):
    pass

class EC2Controller(IPy1Controller):
    _params = (
        config.StringParameter(
            'ec2.config',
            'EC2 config file',
            default=''
        ),
        config.IntParameter(
            'ec2.min_count',
            'Minimum number of EC2 instances to create (default=1).',
            default=1
        ),
        config.IntParameter(
            'ec2.max_count',
            """Maximum number of EC2 instances to create 
            (default=0 means the same number as ec2.min_count).""",
            default=0
        )
    )

    def __init__(self, **options):
        config.setparams(self, options)
        self.ec2 = ec2ipy1.EC2Cluster(self.config)

    def start(self):
        self.ec2.create_instances(self.min_count, self.max_count)
        self.ec2.start_ipython1(engine_on_controller=True)
        self.ipy1taskcontroller = IPy1Controller(self.ec2.task_controller_url) 

    def stop(self):
        self.ec2.terminate_instances()

    def submit(self, tasks):
        self.ipy1taskcontroller.submit(tasks)

    def retrieve(self, deferred_results):
        self.ipy1taskcontroller.retrieve(deferred_results)

    def run(self, tasks):
        self.ipy1taskcontroller.run(tasks)

