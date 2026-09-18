"""CPU tests of task-language memory scope and malformed-artifact rejection."""
import copy
import unittest
import torch

from openpi.zeva.behavior_effect import SCHEMA
from openpi.zeva.behavior_effect_policy import TaskLanguageMemory
from openpi.zeva.retrieval import CausalRetrievalHead


class MemoryTests(unittest.TestCase):
    def setUp(self):
        head = CausalRetrievalHead(input_dim=4,output_dim=2,hidden_dim=8,dropout=0)
        self.retrieval = {"source_feature":"task_language", "task_names":["a","b","outside"],
                          "model_state_dict":head.state_dict(),
                          "task_prototypes":torch.tensor([[.8,.6],[-1.,0.],[1.,0.]])}
        self.bank = {"schema":SCHEMA+"-artifacts", "bank_subset":"train", "tasks":["a","b"],
                     "keys":torch.randn(6,128), "values":torch.randn(6,256), "task_ids":torch.tensor([0,0,0,1,1,1])}

    def test_scope_has_no_true_task_input(self):
        memory = TaskLanguageMemory(self.bank,self.retrieval)
        # Unrestricted classifier would choose 'outside'. The known benchmark
        # scope, not a true task label, restricts it to a/b.
        memory.head.forward = lambda x: torch.tensor([[1.,0.]]).expand(len(x),-1)
        output = memory(torch.zeros(2,4))
        self.assertEqual(output.shape,(2,256))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual([row["task"] for row in memory.last_diagnostics],["a","a"])

    def test_bad_memory_is_rejected(self):
        for mutate in (lambda x:x.update(bank_subset="validation"),
                       lambda x:x.update(values=torch.ones(5,256)),
                       lambda x:x.update(task_ids=torch.ones(6,dtype=torch.long)),
                       lambda x:x["keys"].fill_(float("nan"))):
            bank = copy.deepcopy(self.bank)
            mutate(bank)
            with self.assertRaises(ValueError):
                TaskLanguageMemory(bank,self.retrieval)


if __name__ == "__main__":
    unittest.main()
