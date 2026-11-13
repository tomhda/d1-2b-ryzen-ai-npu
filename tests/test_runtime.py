import json,pathlib,tempfile,unittest
from unittest.mock import patch
from d1npu2.questions import validate_questions,format_answer
from d1npu2.sessions import vision_fingerprint,complete_context

class RuntimeTests(unittest.TestCase):
    def test_question_validation_and_threshold(self):
        q=validate_questions({'q':{'type':'noul','instructions':'Is it visible?','threshold':.7,'invert':True}})['q']
        result=format_answer('noul',['yes','no'],[.2,.8],q['threshold'],q['invert'])
        self.assertEqual(result,{'type':'noul','noul':.8,'decision':True})
        with self.assertRaises(ValueError):validate_questions({'q':{'type':'text','instructions':'Extract text'}})

    def test_choice_and_score_match_checkpoint_answer(self):
        self.assertEqual(format_answer('choice',['a','b'],[.3,.7]),{'type':'choice','choice':'b','confidence':.7,'probabilities':{'a':.3,'b':.7}})
        result=format_answer('score',['0','1','2'],[.3,.15,.55],legend=['low','mid','high'])
        self.assertEqual(result,{'type':'score','score':2.0,'confidence':.55,'probabilities':{'0':.3,'1':.15,'2':.55},'legend':{'0':'low','1':'mid','2':'high'}})
        self.assertEqual(format_answer('score',['0','1','2'],[.2,.3,.5])['score'],1.0)

    def test_external_weight_change_alters_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'vision').mkdir();model=root/'vision'/'vision_encoder_npu.onnx';model.write_bytes(b'graph')
            data=model.with_name('vision_encoder_npu.onnx_data');data.write_bytes(b'weights')
            (root/'model-manifest.json').write_text(json.dumps({'files':[{'path':'vision/vision_encoder_npu.onnx_data','bytes':7,'sha256':'example'}]}))
            first=vision_fingerprint(model);data.write_bytes(b'changed')
            self.assertNotEqual(first,vision_fingerprint(model))

    def test_incomplete_context_never_gets_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'ep').mkdir();(root/'ep'/'vision.onnx').write_bytes(b'graph')
            with self.assertRaises(RuntimeError):complete_context(root,{})
            self.assertFalse((root/'complete.json').exists())

    def test_missing_image_is_rejected_before_native_model_load(self):
        from d1npu2.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);questions=root/'questions.json'
            questions.write_text(json.dumps({'q':{'type':'noul','instructions':'Is it visible?'}}))
            with patch('d1npu2.engine.D1') as constructor:
                with self.assertRaises(FileNotFoundError):main(['decide','--model','unused','--questions',str(questions),'--image',str(root/'missing.png')])
                constructor.assert_not_called()

if __name__=='__main__':unittest.main()
