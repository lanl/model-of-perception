"""Interval answers, scalar projections, and the whole-field area reference."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from task_app.assets import Assets
from task_app.contracts import INTERVAL_AGGREGATIONS, INTERVAL_MEASURES, OPTIONS, compatibility
from task_app.engine import sample_task
from task_app.export import generate
from task_app.parameter_space import describe


class FixedChoices:
    def __init__(self, measure='width', extremum='minimum', scope='all', threshold=25.):
        self.measure,self.extremum,self.scope,self.threshold=measure,extremum,scope,threshold
        # P: rows [0,2), cols [0,3). Q: rows [1,5), cols [2,4).
        self.boundaries=iter(([0,2],[0,3],[1,5],[2,4]))

    def choice(self, options, *args, **kwargs):
        if isinstance(options,int): return np.array(next(self.boundaries))
        if 'width' in options: return self.measure
        if 'axis_scale' in options: return 'axis_scale'
        if 'non_global' in options: return self.scope
        if 'minimum' in options: return self.extremum
        return 3

    def uniform(self,*args,**kwargs): return self.threshold


class IntervalAreaTests(unittest.TestCase):
    def setUp(self):
        self.dataset=dict(values=np.arange(1.,26.).reshape(5,5),extent=[0,5])

    def sample(self, aggregation, action, selection='field', dataset=None, **choices):
        stages=dict(operation='identity',selection=selection,aggregation=aggregation,action=action)
        return sample_task(dataset or self.dataset,stages,FixedChoices(**choices))[0]

    def test_interval_and_area_contracts(self):
        self.assertNotIn('fraction',OPTIONS['aggregation'])
        self.assertFalse(compatibility('identity','field','fraction','retrieve')[0])
        for agg in INTERVAL_AGGREGATIONS:
            for selection in ('field','region','threshold','extrema','saddles'):
                for action in ('retrieve','difference','ratio'):
                    self.assertTrue(compatibility('identity',selection,agg,action)[0])
                for action in ('compare','coordinates','find','count'):
                    self.assertFalse(compatibility('identity',selection,agg,action)[0])
            for selection in ('point','points'):
                for action in OPTIONS['action']:
                    self.assertFalse(compatibility('identity',selection,agg,action)[0])
            for selection in ('regions','inside_outside'):
                for action in ('compare','difference','ratio'):
                    self.assertTrue(compatibility('identity',selection,agg,action)[0])
            self.assertTrue(compatibility('identity','windows',agg,'find')[0])
        for action in ('difference','ratio'):
            self.assertTrue(compatibility('identity','region','area',action)[0])
            self.assertFalse(compatibility('identity','region','mean',action)[0])
            self.assertFalse(compatibility('identity','windows','area',action)[0])

    def test_single_interval_answers_and_zero_denominators(self):
        dataset=dict(values=np.array([[2.,5.,9.],[4.,3.,8.],[6.,7.,2.]]),extent=[-1.5,1.5])
        for agg,pair,difference,ratio in [('range',[2.,9.],7.,4.5),
                                        ('extent_x',[-1.,1.],2.,-1.),('extent_y',[-1.,1.],2.,-1.)]:
            task=self.sample(agg,'retrieve',dataset=dataset)
            self.assertEqual(task['ground_truth'],pair)
            self.assertEqual(task['answer_spec']['type'],'number_pair')
            self.assertEqual(task['answer_spec']['order'],['minimum','maximum'])
            self.assertEqual(task['parameters'],{})
            self.assertIn('Return [minimum, maximum]',task['prompt'])
            self.assertAlmostEqual(self.sample(agg,'difference',dataset=dataset)['ground_truth'],difference)
            task=self.sample(agg,'ratio',dataset=dataset)
            self.assertEqual(task['ground_truth'],ratio)
            self.assertNotIn('comparison_reference',task['context'])
        zero_min=dict(values=dataset['values']-2,extent=[-.5,2.5])
        for agg in INTERVAL_AGGREGATIONS:
            task=self.sample(agg,'ratio',dataset=zero_min)
            self.assertEqual(task['ground_truth'],'undefined')
        flat=dict(values=np.full((3,3),3.),extent=[0,3])
        self.assertEqual(self.sample('range','retrieve',dataset=flat)['ground_truth'],[3.,3.])
        self.assertEqual(self.sample('range','difference',dataset=flat)['ground_truth'],0.)
        self.assertEqual(self.sample('range','ratio',dataset=flat)['ground_truth'],1.)
        with self.assertRaisesRegex(ValueError,'nonempty'):
            self.sample('range','retrieve',selection='threshold',threshold=25.)

    def test_object_intervals_use_values_or_locations(self):
        field=np.zeros((7,7));field[1:3,1:3]=10;field[5,5]=6
        extrema=dict(values=field,extent=[0,7])
        for agg,expected in [('range',[6.,10.]),('extent_x',[1.5,5.5]),('extent_y',[1.5,5.5])]:
            task=self.sample(agg,'retrieve','extrema',extrema,extremum='maximum')
            self.assertEqual(task['ground_truth'],expected)
            self.assertEqual(task['context']['selected_object_count'],2)
            # A four-cell plateau is one object, so a global-only selection is a singleton.
            with self.assertRaisesRegex(ValueError,'two samples'):
                self.sample(agg,'retrieve','extrema',extrema,extremum='maximum',scope='global')
        saddle=dict(values=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]),extent=[0,3])
        for agg,expected in [('range',[.5,.5]),('extent_x',[1.,2.]),('extent_y',[1.,2.])]:
            self.assertEqual(self.sample(agg,'retrieve','saddles',saddle)['ground_truth'],expected)
        with self.assertRaisesRegex(ValueError,'two samples'):
            self.sample('extent_x','retrieve','saddles',dict(values=np.zeros((3,3)),extent=[0,3]))

    def test_two_regions_project_the_same_measure(self):
        pairs={'range':[(1.,8.),(8.,24.)],
               'extent_x':[(.5,2.5),(2.5,3.5)],'extent_y':[(.5,1.5),(1.5,4.5)]}
        for agg,(p,q) in pairs.items():
            for measure in INTERVAL_MEASURES:
                a,b=([v[1]-v[0] for v in (p,q)] if measure=='width' else
                     [v[0 if measure=='minimum' else 1] for v in (p,q)])
                expected={'compare':'Equal' if a==b else 'P' if a>b else 'Q',
                          'difference':abs(a-b),'ratio':max(a,b)/min(a,b)}
                for action,answer in expected.items():
                    task=self.sample(agg,action,'regions',measure=measure)
                    self.assertEqual(task['ground_truth'],answer)
                    self.assertEqual(task['parameters']['interval_measure'],measure)
                    self.assertIn(f'Use: {measure}',task['parameter_summary'])
                    self.assertNotIn('comparison_reference',task['context'])

    def test_window_projections_and_ties(self):
        field=self.dataset['values'].copy();field[0,0]=1000.
        dataset=dict(values=field,extent=[0,5])
        x,y=np.meshgrid(np.arange(5)+.5,np.arange(5)+.5)
        for agg,source in [('range',field),('extent_x',x),('extent_y',y)]:
            for measure in INTERVAL_MEASURES:
                measurements={}
                for row in range(3):
                    for col in range(3):
                        block=source[row:row+3,col:col+3]
                        measurements[(row+1,col+1)]=(block.max()-block.min() if measure=='width' else
                                                     block.min() if measure=='minimum' else block.max())
                for extremum in ('minimum','maximum'):
                    best=(min if extremum=='minimum' else max)(measurements.values())
                    expected={cell for cell,value in measurements.items() if value==best}
                    task=self.sample(agg,'find','windows',dataset,measure=measure,extremum=extremum)
                    self.assertEqual({tuple(c) for c in task['ground_truth']['accepted_cells_row_col']},expected)
                    self.assertEqual(task['parameters']['find_extremum'],extremum)
                    self.assertEqual(task['parameters']['interval_measure'],measure)
        # Equal spatial spans must remain tied even with non-exact decimal spacing.
        for agg in ('extent_x','extent_y'):
            task=self.sample(agg,'find','windows',dict(values=field,extent=[-.015,.135]),measure='width')
            self.assertEqual(len(task['ground_truth']['accepted_xy']),9)

    def test_parameter_counts_include_projection_choice(self):
        data=dict(values=np.zeros((3,3)))
        for agg in INTERVAL_AGGREGATIONS:
            for selection,action,count in [('field','retrieve',1),('regions','compare',1296*3),('windows','find',12)]:
                stages=dict(operation='identity',selection=selection,aggregation=agg,action=action)
                space=describe(stages,data)
                self.assertEqual(space['discrete_settings'],str(count))
                self.assertEqual(space['continuous_dimensions'],0)

    def test_area_uses_whole_field_only_for_a_single_group(self):
        dataset=dict(self.dataset,extent=[0,10]) # 25 cells, each of area 4.
        for action,expected in [('retrieve',24.),('ratio',.24),('difference',76.)]:
            task=self.sample('area',action,'region',dataset)
            self.assertEqual(task['ground_truth'],expected)
            self.assertEqual(set(task['parameters']),{'rectangles'})
            if action!='retrieve':
                self.assertEqual(task['context']['reference_area'],100.)
                self.assertEqual(task['context']['comparison_reference'],'whole_field_area')
                self.assertIn('whole field area',task['prompt'])
        for action,empty,full in [('ratio',0.,1.),('difference',100.,0.)]:
            self.assertEqual(self.sample('area',action,'threshold',dataset,threshold=25.)['ground_truth'],empty)
            self.assertEqual(self.sample('area',action,'field',dataset)['ground_truth'],full)
        # With two groups the second group, rather than the full field, is the operand.
        for selection,ratio,difference in [('regions',32/24,8.),('inside_outside',76/24,52.)]:
            for action,expected in [('ratio',ratio),('difference',difference)]:
                task=self.sample('area',action,selection,dataset)
                self.assertEqual(task['ground_truth'],expected)
                self.assertNotIn('comparison_reference',task['context'])
        space=describe(dict(operation='identity',selection='field',aggregation='area',action='ratio'),dataset)
        self.assertEqual(space['discrete_settings'],'1') # The fixed reference adds no sampled parameter.

    def test_interval_and_area_exports(self):
        assets=Assets()
        config=dict(datasets=['s01'],colormaps=['gray'],operation=['identity'],selection=['field'],
                    aggregation=list(INTERVAL_AGGREGATIONS),action=['retrieve'],count=3,seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'intervals'
            result=generate(config,assets,output)
            self.assertEqual(result['manifest']['actual_count'],3)
            self.assertEqual({r['aggregation'] for r in result['records']},set(INTERVAL_AGGREGATIONS))
            rows=list(csv.DictReader((output/'tasks.csv').read_text().splitlines()))
            for row,record in zip(rows,result['records']):
                self.assertEqual(json.loads(row['ground_truth']),record['ground_truth'])
                self.assertEqual(json.loads(row['answer_spec'])['type'],'number_pair')
                self.assertEqual(len(record['ground_truth']),2)
            with zipfile.ZipFile(output/'everything.zip') as archive:
                self.assertIn('tasks.md',archive.namelist())
                self.assertIn(result['records'][0]['image_ref'],archive.namelist())
            result=generate(dict(config,aggregation=['area'],action=['ratio'],count=1),assets,Path(tmp)/'area')
            self.assertEqual(result['records'][0]['ground_truth'],1.)
            self.assertEqual(result['records'][0]['context']['comparison_reference'],'whole_field_area')


if __name__=='__main__': unittest.main()
