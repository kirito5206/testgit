import unittest
from types import SimpleNamespace
from voice_dataset.core import windows, clip_id, merge_short, add_context, split_asr_segment

class HelpersTest(unittest.TestCase):
    def test_windows(self):
        ws=windows(251,120,4)
        self.assertEqual(len(ws),3)
        self.assertEqual(ws[1].read_start,116)
        self.assertEqual(ws[-1].core_end,251)
    def test_ids(self):
        self.assertEqual(clip_id("a",1,3),clip_id("a",1,3))
        self.assertNotEqual(clip_id("a",1,3),clip_id("b",1,3))
    def test_split(self):
        words=[SimpleNamespace(start=i,end=i+0.8,word=f"w{i}") for i in range(8)]
        seg=SimpleNamespace(start=0,end=8,text="test",words=words)
        chunks=split_asr_segment(seg,3)
        self.assertEqual(len(chunks),3)
        self.assertTrue(all(b-a<=3 for a,b,_ in chunks))
    def test_merge_and_context(self):
        items=[dict(start=0,end=1,text="你好",language="zh"),dict(start=1.2,end=2.5,text="世界",language="zh"),dict(start=4,end=7,text="结束",language="zh")]
        output=merge_short(items,2,15)
        self.assertEqual(output[0]["text"],"你好世界")
        add_context(output)
        self.assertEqual(output[1]["context_before"],["你好世界"])

if __name__=="__main__": unittest.main()
