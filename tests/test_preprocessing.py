import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from mice_recog.src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """Set up sample data for tests"""
        self.data_dir = Path('./')
        self.output_dir = Path('./test_output')
        self.preprocessor = DataPreprocessor(self.data_dir, self.output_dir)

        # Sample tracking data
        tracking_data = {
            'video_frame': [0, 0, 0, 0, 1, 1, 1, 1],
            'mouse_id': [1, 1, 2, 2, 1, 1, 2, 2],
            'bodypart': ['nose', 'tail_base', 'nose', 'tail_base', 'nose', 'tail_base', 'nose', 'tail_base'],
            'x': [10, 20, 100, 120, 11, 21, 101, 121],
            'y': [10, 20, 100, 120, 11, 21, 101, 121]
        }
        self.tracking_df = pd.DataFrame(tracking_data)

        # Sample metadata
        self.metadata = pd.Series({'body_parts_tracked': ['nose', 'tail_base']})

    def test_create_raw_pose_features(self):
        """Test the _create_raw_pose_features method"""
        raw_pose = self.preprocessor._create_raw_pose_features(self.tracking_df, self.metadata)

        # Check shape
        self.assertEqual(raw_pose.shape, (4, 8))

        # Check columns
        expected_cols = ['video_frame', 'mouse_id', 'x_nose', 'x_tail_base', 'y_nose', 'y_tail_base', 'vis_nose', 'vis_tail_base']
        self.assertListEqual(sorted(raw_pose.columns.tolist()), sorted(expected_cols))

        # Check visibility flags
        self.assertTrue(np.all(raw_pose['vis_nose'] == 1.0))
        self.assertTrue(np.all(raw_pose['vis_tail_base'] == 1.0))

    def test_create_mouse_center_features(self):
        """Test the _create_mouse_center_features method"""
        raw_pose = self.preprocessor._create_raw_pose_features(self.tracking_df, self.metadata)
        mouse_center = self.preprocessor._create_mouse_center_features(raw_pose, self.metadata)

        # Check shape
        self.assertEqual(mouse_center.shape, (4, 8))

        # Check center calculations
        self.assertAlmostEqual(mouse_center.loc[mouse_center['mouse_id'] == 1, 'cx'].iloc[0], 15.0)
        self.assertAlmostEqual(mouse_center.loc[mouse_center['mouse_id'] == 2, 'cy'].iloc[0], 110.0)

        # Check velocity calculations
        self.assertAlmostEqual(mouse_center.loc[mouse_center['mouse_id'] == 1, 'vx'].iloc[1], 1.0)
        self.assertAlmostEqual(mouse_center.loc[mouse_center['mouse_id'] == 2, 'vy'].iloc[1], 1.0)

    def test_create_pairwise_features(self):
        """Test the _create_pairwise_features method"""
        raw_pose = self.preprocessor._create_raw_pose_features(self.tracking_df, self.metadata)
        mouse_center = self.preprocessor._create_mouse_center_features(raw_pose, self.metadata)
        pairwise = self.preprocessor._create_pairwise_features(mouse_center)

        # Check shape (2 frames * 3 pairs per frame = 6)
        self.assertEqual(pairwise.shape, (6, 10))

        # Check distance calculation for a specific pair
        pair_1_2_frame_0 = pairwise[(pairwise['agent_id'] == 2) & (pairwise['target_id'] == 1) & (pairwise['video_frame'] == 0)]
        self.assertAlmostEqual(pair_1_2_frame_0['dist'].iloc[0], np.sqrt((110-15)**2 + (110-15)**2))

if __name__ == '__main__':
    unittest.main()
