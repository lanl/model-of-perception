import base64
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from task_app.assets import Assets, load_colors, load_grid
from task_app.contracts import OPTIONS, compatibility, combinations, combination_id, Shape, aggregation_output
from task_app.critical_points import bilinear_saddles
from task_app.engine import sample_task, render
from task_app.interpolation import scalar_at, scalar_raster
from task_app.export import generate, task_fingerprint
from task_app.parameter_space import describe, summarize

class AppTests(unittest.TestCase):
    def setUp(self): self.assets=Assets()

    def test_contract_composition(self):
        self.assertTrue(compatibility('identity','regions','range','compare')[0])
        self.assertFalse(compatibility('identity','regions','mean','find')[0])
        self.assertFalse(compatibility('identity','field','mean','compare')[0])
        self.assertTrue(compatibility('identity','field','components','count')[0])
        self.assertTrue(compatibility('identity','extrema','mean','retrieve')[0])
        self.assertTrue(compatibility('identity','saddles','mean','retrieve')[0])
        self.assertTrue(compatibility('identity','saddles','identity','coordinates')[0])
        self.assertTrue(compatibility('identity','extrema','identity','coordinates')[0])
        self.assertFalse(compatibility('identity','saddles','mean','coordinates')[0])
        self.assertFalse(compatibility('identity','extrema','mean','find')[0])
        for operation in OPTIONS['operation']:
            for aggregation in OPTIONS['aggregation']:
                for action in OPTIONS['action']:
                    self.assertEqual(compatibility(operation,'extrema',aggregation,action)[0],
                                     compatibility(operation,'saddles',aggregation,action)[0],
                                     (operation,aggregation,action))
        with self.assertRaises(ValueError): aggregation_output(Shape('objects'), 'mean')
        self.assertTrue(compatibility('identity','point','identity','retrieve')[0])
        self.assertTrue(compatibility('identity','points','identity','compare')[0])
        # Two marked points are two singleton groups, not one two-value group.
        for selection in ('point','points'):
            for aggregation in OPTIONS['aggregation']:
                if aggregation == 'identity': continue
                for action in OPTIONS['action']:
                    self.assertFalse(compatibility('identity',selection,aggregation,action)[0])

    def test_sampled_singleton_region_rejected(self):
        class OneCellRegion:
            def choice(self,*args,**kwargs): return np.array([0,1])
        for aggregation in ('mean','std','range','extent_x','extent_y','minimum','maximum','area'):
            with self.assertRaisesRegex(ValueError,'at least two samples'):
                sample_task(self.assets.datasets['s02'],
                            dict(operation='identity',selection='region',aggregation=aggregation,action='retrieve'),
                            OneCellRegion())

    def test_extrema_mean_weights_plateaus_once(self):
        field=np.zeros((7,7));field[1:3,1:3]=10;field[5,5]=6
        dataset=dict(values=field,extent=[0,7])
        class FixedExtrema:
            def __init__(self,kind='maximum',scope='all'):self.kind,self.scope=kind,scope
            def choice(self,choices):return self.kind if self.kind in choices else self.scope
        stages=dict(operation='identity',selection='extrema',aggregation='mean',action='retrieve')
        task,*_=sample_task(dataset,stages,FixedExtrema())
        self.assertEqual(task['ground_truth'],8.) # Not (4*10+6)/5: equal weight per plateau.
        self.assertEqual(task['context']['selected_object_count'],2)
        self.assertEqual(task['parameters'],{'extremum':'maximum','scope':'all'})
        self.assertIn('Extremum: maximum',task['parameter_summary'])
        self.assertIn('equal weight',task['prompt'])
        task,*_=sample_task(dict(dataset,values=-field),stages,FixedExtrema('minimum'))
        self.assertEqual(task['ground_truth'],-8.)
        for scope in ('global','non_global'):
            with self.assertRaisesRegex(ValueError,'at least two samples'):
                sample_task(dataset,stages,FixedExtrema(scope=scope))
        for action,expected in [('count',2),('exists',True),('difference',4.),('ratio',10/6)]:
            task,*_=sample_task(dataset,dict(stages,aggregation='identity',action=action),FixedExtrema())
            self.assertEqual(task['ground_truth'],expected)
        buf=io.BytesIO();np.save(buf,field,allow_pickle=False)
        ident=self.assets.add_grid('Unequal plateaus',buf.getvalue(),'plateaus.npy',[0,7])
        config=dict(datasets=[ident],colormaps=['gray'],operation=['identity'],selection=['extrema'],
                    aggregation=['mean'],action=['retrieve'],count=1,seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            result=generate(config,self.assets,Path(tmp)/'mean')
            self.assertEqual(result['manifest']['actual_count'],1)
            self.assertEqual(result['records'][0]['ground_truth'],8.)

    def test_extrema_coordinate_answers(self):
        field=np.zeros((7,7));field[1:3,1:3]=10;field[5,5]=6
        class FixedExtrema:
            def __init__(self,kind,scope): self.kind,self.scope=kind,scope
            def choice(self,choices): return self.kind if self.kind in choices else self.scope
        stages=dict(operation='identity',selection='extrema',aggregation='identity',action='coordinates')
        plateau={(1,1),(1,2),(2,1),(2,2)}
        for kind,values in [('maximum',field),('minimum',-field)]:
            for scope,expected in [('all',plateau|{(5,5)}),('global',plateau),('non_global',{(5,5)})]:
                task,*_=sample_task(dict(values=values,extent=[0,7]),stages,FixedExtrema(kind,scope))
                self.assertEqual(task['answer_spec']['type'],'coordinates')
                self.assertEqual({tuple(cell) for cell in task['ground_truth']['accepted_cells_row_col']},expected)
                self.assertEqual({tuple(point) for point in task['ground_truth']['accepted_xy']},
                                 {(c+.5,r+.5) for r,c in expected})
                self.assertIn(kind,task['prompt'])
                self.assertIn('any cell center is accepted',task['prompt'])
                if scope=='non_global': self.assertIn('not global',task['prompt'])
        with self.assertRaisesRegex(ValueError,'valid location'):
            sample_task(dict(values=np.zeros((3,3)),extent=[0,3]),stages,FixedExtrema('maximum','non_global'))
        # Multiple cells in one plateau still count as one object for scalar comparisons.
        for action in ('difference','ratio'):
            task,*_=sample_task(dict(values=field,extent=[0,7]),dict(stages,action=action),FixedExtrema('maximum','global'))
            self.assertEqual(task['ground_truth'],'undefined')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'extrema-coordinates'
            result=generate(dict(datasets=['terrain_100'],colormaps=['gray'],count=1,seed=42,
                                 **{key:[value] for key,value in stages.items()}),self.assets,output)
            self.assertEqual(result['manifest']['actual_count'],1)
            record=result['records'][0]
            self.assertTrue(record['ground_truth']['accepted_xy'])
            row=list(csv.DictReader(io.StringIO((output/'tasks.csv').read_text())))[0]
            self.assertEqual(json.loads(row['ground_truth']),record['ground_truth'])
            self.assertIn(record['prompt'],(output/'tasks.md').read_text())

    def test_bilinear_saddles_and_actions(self):
        x,y=np.meshgrid([-1.,0.,1.],[-1.,0.,1.])
        field=(x-.2)*(y+.3)+4
        points=bilinear_saddles(field,[-1.5,1.5])
        self.assertEqual(len(points),1)
        np.testing.assert_allclose(points[0]['xy'],[.2,-.3],atol=1e-14)
        self.assertAlmostEqual(points[0]['value'],4.)
        self.assertAlmostEqual(points[0]['value'],float(scalar_at(field,[-1.5,1.5],points[0]['xy'])))
        for nonsaddle in (np.zeros((3,3)), x+y, x*x+y*y, x*y):
            self.assertEqual(bilinear_saddles(nonsaddle,[-1.5,1.5]),[])
        dataset=dict(values=field,extent=[-1.5,1.5])
        for action,expected in [('count',1),('exists',True)]:
            task,*_=sample_task(dataset,dict(operation='identity',selection='saddles',aggregation='identity',action=action),np.random.default_rng(1))
            self.assertEqual(task['ground_truth'],expected)
        task,*_=sample_task(dataset,dict(operation='identity',selection='saddles',aggregation='identity',action='find'),np.random.default_rng(1))
        np.testing.assert_allclose(task['ground_truth']['accepted_xy'],[[.2,-.3]],atol=1e-14)
        stages=dict(operation='identity',selection='saddles',aggregation='mean',action='retrieve')
        with self.assertRaisesRegex(ValueError,'at least two samples'):
            sample_task(dataset,stages,np.random.default_rng(1))
        checker=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        task,*_=sample_task(dict(values=checker,extent=[0,3]),stages,np.random.default_rng(1))
        self.assertEqual(task['context']['selected_object_count'],4)
        self.assertEqual(task['ground_truth'],.5)

    def test_saddle_coordinate_answers(self):
        checker=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        dataset=dict(values=checker,extent=[0,3])
        stages=dict(operation='identity',selection='saddles',aggregation='identity',action='coordinates')
        task,*_=sample_task(dataset,stages,np.random.default_rng(42))
        self.assertEqual(task['answer_spec']['type'],'coordinates')
        np.testing.assert_allclose(task['ground_truth']['accepted_xy'],[[1,1],[2,1],[1,2],[2,2]])
        self.assertIn('Return one [x, y] pair',task['prompt'])
        with self.assertRaisesRegex(ValueError,'valid location'):
            sample_task(dict(dataset,values=np.zeros((3,3))),stages,np.random.default_rng(42))
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(datasets=['terrain_100'],colormaps=['gray'],count=1,seed=42,
                        **{key:[value] for key,value in stages.items()})
            result=generate(config,self.assets,Path(tmp)/'coordinates')
            self.assertEqual(result['manifest']['actual_count'],1)
            self.assertEqual(result['records'][0]['answer_spec']['type'],'coordinates')
            self.assertTrue(result['records'][0]['ground_truth']['accepted_xy'])

    def test_saddles_are_an_unpaired_set(self):
        self.assertNotIn('saddle_pair',OPTIONS['selection'])
        self.assertFalse(compatibility('identity','saddles','identity','compare')[0])
        for field,expected in [(np.zeros((3,3)),0),
                               (np.array([[1.,0.,-1.],[0.,0.,0.],[-1.,0.,1.]]),0),
                               (np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]),4)]:
            task,*_=sample_task(dict(values=field,extent=[0,3]),
                                dict(operation='identity',selection='saddles',aggregation='identity',action='count'),
                                np.random.default_rng(42))
            self.assertEqual(task['ground_truth'],expected)

    def test_saddle_set_difference_and_ratio(self):
        varied=np.array([[0.,2.,0.],[2.,0.,4.],[0.,4.,0.]])
        tied=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        x,y=np.meshgrid([.5,1.5,2.5],[.5,1.5,2.5])
        single=(x-1)*(y-1)+4
        examples=[(varied,1.,2.),(tied,0.,1.),(varied-1,1.,'undefined'),
                  (varied-3,1.,.5),(np.zeros((3,3)),'undefined','undefined'),
                  (single,'undefined','undefined')]
        for field,difference,ratio in examples:
            for action,expected in [('difference',difference),('ratio',ratio)]:
                stages=dict(operation='identity',selection='saddles',aggregation='identity',action=action)
                self.assertTrue(compatibility(**stages)[0])
                task,*_=sample_task(dict(values=field,extent=[0,3]),stages,np.random.default_rng(42))
                self.assertEqual(task['ground_truth'],expected)
                self.assertEqual(task['answer_spec']['type'],'number_or_undefined')
                self.assertEqual(task['answer_spec']['undefined_token'],'undefined')
        self.assertEqual(OPTIONS['selection']['extrema'],'Extrema')
        self.assertFalse(compatibility('identity','saddles','identity','compare')[0])
        # The same capability rule supports other scalar-valued object sets.
        self.assertTrue(compatibility('identity','extrema','identity','difference')[0])
        self.assertFalse(compatibility('identity','field','components','difference')[0])
        buf=io.BytesIO();np.save(buf,np.zeros((3,3)),allow_pickle=False)
        ident=self.assets.add_grid('No saddles',buf.getvalue(),'flat.npy',[0,3])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'undefined'
            config=dict(datasets=[ident],colormaps=['gray'],operation=['identity'],selection=['saddles'],
                        aggregation=['identity'],action=['difference'],count=1,seed=42)
            result=generate(config,self.assets,output)
            self.assertEqual(result['manifest']['actual_count'],1)
            self.assertEqual(result['records'][0]['ground_truth'],'undefined')
            self.assertEqual(list(csv.DictReader(io.StringIO((output/'tasks.csv').read_text())))[0]['ground_truth'],'undefined')
            self.assertEqual(json.loads((output/'tasks.json').read_text())[0]['ground_truth'],'undefined')

    def test_explicit_combination_exclusions(self):
        selected=dict(operation=['identity'],selection=['field'],aggregation=['mean','maximum'],action=['retrieve'])
        original=combinations(selected)
        excluded=combination_id(original[0])
        selected['excluded_combinations']=[excluded]
        self.assertEqual(combinations(selected),original[1:])
        self.assertEqual(combinations(selected,include_excluded=True),original)
        selected['aggregation']=['maximum']
        self.assertEqual(combinations(selected),original[1:])
        selected['aggregation']=['mean','maximum']
        self.assertEqual(combinations(selected),original[1:])
        selected['excluded_combinations']=[combination_id(c) for c in original]
        self.assertEqual(combinations(selected),[])
        selected['excluded_combinations']=[]
        self.assertEqual(combinations(selected),original)
        for invalid in ['not-a-combination',['identity|point|mean|retrieve'],[None]]:
            selected['excluded_combinations']=invalid
            with self.assertRaises(ValueError): combinations(selected)

    def test_interpolation_and_source_resolution(self):
        # A bilinear saddle: its midpoint is .5, with no overshoot or RGBA blend.
        field=np.array([[0.,1.],[1.,0.]])
        points=np.array([[.5,.5],[1.5,.5],[1.,1.],[.75,.75],[0.,0.],[2.,2.]])
        np.testing.assert_allclose(scalar_at(field,[0,2],points),[0,1,.5,.375,0,0])
        raster=scalar_raster(field,[0,2],resolution=4)
        self.assertEqual(raster.shape,(4,4))
        self.assertGreaterEqual(raster.min(),0)
        self.assertLessEqual(raster.max(),1)
        with self.assertRaises(ValueError): scalar_at(field,[0,2],[-.01,0])
        self.assertNotIn('s00',self.assets.datasets)
        terrain=self.assets.datasets['terrain_100']
        self.assertEqual(terrain['values'].shape,(100,100))
        np.testing.assert_equal(scalar_at(terrain['values'],terrain['extent'],[[terrain['extent'][0]]*2,[terrain['extent'][1]]*2]),
                                [terrain['values'][0,0],terrain['values'][-1,-1]])
        stages=dict(operation='identity',selection='point',aggregation='identity',action='retrieve')
        record,values,overlays=sample_task(terrain,stages,np.random.default_rng(42))
        self.assertEqual(record['ground_truth'],float(scalar_at(values,terrain['extent'],overlays[0][2])))
        self.assertEqual(record['context']['interpolation']['stage'],'scalar_before_colormap')
        self.assertEqual(set(record['parameters']),{'cells'})
        x,y=overlays[0][2]
        self.assertEqual(record['parameter_summary'],[f'Point P: x = {x:.10g}, y = {y:.10g}'])

    def test_known_answers_and_ties(self):
        dataset=dict(values=np.array([[1,1,3],[2,3,3],[1,2,3.]]),extent=[0,3])
        for aggregation,expected in [('mean',19/9),('std',np.std(dataset['values'])),('minimum',1),('maximum',3),('area',9)]:
            record,*_=sample_task(dataset,dict(operation='identity',selection='field',aggregation=aggregation,action='retrieve'),np.random.default_rng(1))
            self.assertAlmostEqual(record['ground_truth'],expected)
            self.assertEqual(record['parameters'],{})
            self.assertEqual(record['parameter_summary'],['None beyond the selected stages.'])
        record,*_=sample_task(dataset,dict(operation='identity',selection='field',aggregation='maximum',action='find'),np.random.default_rng(1))
        self.assertEqual(len(record['ground_truth']['accepted_xy']),4)
        self.assertEqual(record['ground_truth']['accepted_xy'][0],[2.5,.5])

    def test_gradient_tasks_display_original_field(self):
        field=np.array([[0.,1.,4.],[2.,3.,8.],[4.,9.,16.]])+100
        gradient=np.array([[2.,3.,4.],[2.,6.,8.],[5.,7.,8.]])
        dataset=dict(values=field,extent=[0,3])
        stages=dict(operation='gradient',selection='field',aggregation='mean',action='retrieve')
        record,analysis,overlays=sample_task(dataset,stages,np.random.default_rng(42))
        np.testing.assert_array_equal(analysis,gradient)
        self.assertEqual(record['ground_truth'],5.)
        self.assertEqual(record['context']['color_range'],[100.,116.])
        self.assertEqual(record['context']['analysis_range'],[2.,8.])
        self.assertEqual(record['context']['display_field'],'original_scalar_field')
        self.assertEqual(record['context']['analysis_field'],'gradient_magnitude')
        self.assertIn('gradient magnitude of the displayed scalar field',record['prompt'])
        self.assertNotIn('colors show gradient',record['prompt'])
        class FixedThreshold:
            def uniform(self,lower,upper):
                self.bounds=(lower,upper)
                return 5.
        rng=FixedThreshold()
        filtered,*_=sample_task(dataset,dict(stages,selection='threshold',aggregation='identity',action='count'),rng)
        self.assertEqual(rng.bounds,(2.,8.))
        self.assertEqual(filtered['ground_truth'],4)
        # Verify the exported PNG, including its color scale, uses the source
        # field while the exported answer still comes from the gradient.
        buf=io.BytesIO();np.save(buf,field,allow_pickle=False)
        ident=self.assets.add_grid('Offset field',buf.getvalue(),'offset.npy',[0,3])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'gradient'
            result=generate(dict(datasets=[ident],colormaps=['gray'],count=1,seed=42,
                                 **{key:[value] for key,value in stages.items()}),self.assets,output)
            task=result['records'][0]
            self.assertEqual(task['ground_truth'],5.)
            expected=render(field,overlays,dataset['extent'],self.assets.cmap('gray'))
            self.assertEqual((output/task['image_ref']).read_bytes(),expected)
            row=list(csv.DictReader(io.StringIO((output/'tasks.csv').read_text())))[0]
            self.assertEqual(json.loads(row['context'])['color_range'],[100.,116.])

    def test_upload_validation(self):
        np.testing.assert_equal(load_grid('x.csv',b'1,2,3\n4,5,6\n7,8,9'),np.arange(1,10).reshape(3,3))
        for data in [b'1,2\n3,4',b'1,2,3\n4,nan,6\n7,8,9']:
            with self.assertRaises(ValueError): load_grid('x.csv',data)
        self.assertEqual(load_colors('x.json',b'["#000000","#ffffff"]')[-1],[1,1,1,1])
        with self.assertRaises(ValueError): load_colors('x.json',b'["transparent","red"]')

    def test_all_contracts_execute_or_fail_data_validation(self):
        # Catch shape/arity implementation errors across every inferred combination.
        for i,stages in enumerate(combinations({k:list(v) for k,v in OPTIONS.items()})):
            try:
                result,*_=sample_task(self.assets.datasets['s02'],stages,np.random.default_rng(i))
                json.dumps(result,allow_nan=False)
                self.assertTrue(result['prompt'])
                self.assertNotIn('scoring',result)
            except ValueError as exc:
                self.assertTrue(any(word in str(exc) for word in ('nonempty','positive','constant','valid location','two cells','two samples')), (stages,str(exc)))

    def test_exports_seed_and_shortfall(self):
        config=dict(datasets=['terrain_100'],colormaps=['viridis'],operation=['identity'],selection=['regions'],aggregation=['range'],action=['compare'],count=2,seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            a=generate(config,self.assets,Path(tmp)/'a'); b=generate(config,self.assets,Path(tmp)/'b')
            self.assertEqual(a['records'],b['records'])
            rows=list(csv.DictReader(io.StringIO((Path(tmp)/'a/tasks.csv').read_text())))
            self.assertEqual(len(rows),2)
            self.assertNotIn('scoring',rows[0])
            self.assertEqual(set(json.loads(rows[0]['parameters'])),{'rectangles','interval_measure'})
            self.assertEqual(json.loads(rows[0]['context']),a['records'][0]['context'])
            record=a['records'][0]
            self.assertEqual(record['parameter_summary'],[
                f'Box {label} bounds: x = {x0:.10g} to {x1:.10g}; y = {y0:.10g} to {y1:.10g}'
                for label,(x0,x1,y0,y1) in zip(('P','Q'),record['parameters']['rectangles'])]+
                [f'Use: {record["parameters"]["interval_measure"]}'])
            markdown=(Path(tmp)/'a/tasks.md').read_text()
            self.assertIn('Sampled parameters',markdown)
            self.assertNotIn('Scoring',markdown)
            self.assertNotIn('scoring',record)
            self.assertNotIn('comparison_tolerance',record['context'])
            self.assertIn('comparison_equality_epsilon',record['context'])
            self.assertIn('<br>'.join(record['parameter_summary']),markdown)
            self.assertNotIn('scalar_before_colormap',markdown)
            self.assertEqual(json.loads((Path(tmp)/'a/tasks.json').read_text()),a['records'])
            with zipfile.ZipFile(Path(tmp)/'a/everything.zip') as archive:
                self.assertIn('tasks.md',archive.namelist())
                for task in json.loads(archive.read('tasks.json')):
                    self.assertNotIn('scoring',task)
                    self.assertIn('ground_truth',task)
                    self.assertIn('answer_spec',task)
                self.assertIn(rows[0]['image_ref'],archive.namelist())
                self.assertIn('inputs/terrain_100.npy',archive.namelist())
            self.assets.datasets['terrain_100']['values']=np.ones((3,3))
            config.update(selection=['field'],aggregation=['pearson'],action=['retrieve'])
            failed=generate(config,self.assets,Path(tmp)/'failed')
            self.assertEqual(failed['manifest']['actual_count'],0)
            self.assertEqual(len(failed['manifest']['failures']),2)

    def test_duplicates_and_finite_space(self):
        config=dict(datasets=['s01'],colormaps=['gray'],operation=['identity'],selection=['field'],
                    aggregation=['minimum','maximum'],action=['retrieve'],count=3,seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            result=generate(config,self.assets,Path(tmp)/'finite')
            self.assertEqual(result['manifest']['actual_count'],2)
            self.assertEqual({r['aggregation'] for r in result['records']},{'minimum','maximum'})
            self.assertEqual(result['manifest']['failures'][0]['kind'],'uniqueness')
            self.assertGreater(result['manifest']['duplicates_rejected'],0)
            keys=[r['provenance']['task_fingerprint'] for r in result['records']]
            self.assertEqual(len(keys),len(set(keys)))
            config.update(aggregation=['mean'],colormaps=['gray','viridis'],count=2)
            colored=generate(config,self.assets,Path(tmp)/'colors')
            self.assertEqual(len(colored['records']),2)
            self.assertEqual({r['colormap'] for r in colored['records']},{'gray','viridis'})

    def test_parameter_space_dimensions(self):
        data=dict(values=np.zeros((3,3)))
        stages=dict(operation='identity',selection='point',aggregation='identity',action='retrieve')
        self.assertEqual(describe(stages,data)['discrete_settings'],'9')
        stages.update(selection='points',action='compare')
        self.assertEqual(describe(stages,data)['discrete_settings'],'72')
        stages.update(selection='regions',aggregation='mean')
        self.assertEqual(describe(stages,data)['discrete_settings'],'1296')
        stages.update(selection='between_thresholds',aggregation='identity',action='find')
        self.assertEqual(describe(stages,data)['continuous_dimensions'],2)
        for selection in ('threshold','below_threshold'):
            stages['selection']=selection
            self.assertEqual(describe(stages,data)['continuous_dimensions'],1)
        stages.update(selection='windows',aggregation='mean')
        self.assertEqual(describe(stages,data)['discrete_settings'],'4')

    def test_window_size_cues(self):
        class FixedWindow:
            def __init__(self,cue,width=65): self.cue,self.width=cue,width
            def choice(self,options):
                if 'axis_scale' in options: return self.cue
                if 'minimum' in options: return 'minimum'
                return self.width
        dataset=self.assets.datasets['terrain_100']
        stages=dict(operation='identity',selection='windows',aggregation='mean',action='find')
        tasks=[]
        for cue in ('axis_scale','reference_square'):
            task,_,overlays=sample_task(dataset,stages,FixedWindow(cue))
            tasks.append(task)
            self.assertAlmostEqual(task['context']['window_side_length'],1.9695)
            self.assertEqual(task['parameters']['window_size_cue'],cue)
            self.assertEqual(set(task['parameters']),{'window_cells','window_size_cue','find_extremum'})
            self.assertEqual(task['parameter_summary'],[
                'Window width: 65 cells',
                'Size cue: '+('axis scale and ruler' if cue=='axis_scale' else 'reference square R'),
                'Find: minimum'])
            self.assertNotIn('65 × 65 cell',task['prompt'])
            if cue == 'axis_scale':
                self.assertEqual(overlays[0][0],'window_scale')
                self.assertIn('1.9695',task['prompt'])
            else:
                self.assertIn('reference square R',task['prompt'])
                x0,x1,y0,y1=task['context']['window_reference_bounds']
                self.assertAlmostEqual(x1-x0,1.9695)
                self.assertAlmostEqual(y1-y0,1.9695)
                self.assertGreaterEqual(min(x0,y0),dataset['extent'][0])
                self.assertLessEqual(max(x1,y1),dataset['extent'][1])
        self.assertEqual(tasks[0]['ground_truth'],tasks[1]['ground_truth'])
        full,*_=sample_task(dataset,stages,FixedWindow('reference_square',100))
        np.testing.assert_allclose(full['context']['window_reference_bounds'],
                                   [*dataset['extent'],*dataset['extent']])

    def test_threshold_selections(self):
        dataset=dict(values=np.arange(9,dtype=float).reshape(3,3),extent=[0,3])
        class FixedDraws:
            def uniform(self,low,high,size=None):
                return np.array([6.,2.]) if size == 2 else 4.
        for selection,expected,words in [('threshold',4,'strictly above 4'),
                                         ('below_threshold',4,'strictly below 4'),
                                         ('between_thresholds',5,'between 2 and 6 (inclusive)')]:
            stages=dict(operation='identity',selection=selection,aggregation='identity',action='count')
            task,*_=sample_task(dataset,stages,FixedDraws())
            self.assertEqual(task['ground_truth'],expected)
            self.assertIn(words,task['prompt'])
            self.assertNotIn('target_tolerance',task['parameters'])
            if selection == 'between_thresholds':
                self.assertEqual(task['parameters'],{'lower_threshold':2.,'upper_threshold':6.})
                self.assertEqual(task['parameter_summary'],['Lower threshold: 2','Upper threshold: 6'])
            else:
                self.assertEqual(task['parameters'],{'threshold':4.})
                self.assertEqual(task['parameter_summary'],['Threshold: 4'])
        lower,upper=[],[]
        for seed in range(12):
            task,*_=sample_task(dataset,dict(operation='identity',selection='between_thresholds',
                                            aggregation='identity',action='count'),np.random.default_rng(seed))
            p=task['parameters']; lower.append(p['lower_threshold']); upper.append(p['upper_threshold'])
            self.assertLessEqual(0,p['lower_threshold'])
            self.assertLessEqual(p['lower_threshold'],p['upper_threshold'])
            self.assertLessEqual(p['upper_threshold'],8)
        self.assertEqual(len(set(lower)),12)
        self.assertEqual(len(set(upper)),12)
        self.assertGreater(len(set(b-a for a,b in zip(lower,upper))),1)
        self.assertNotIn('target',OPTIONS['selection'])

if __name__=='__main__':unittest.main()
