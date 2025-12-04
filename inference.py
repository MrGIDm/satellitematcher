import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ssl
import kornia as K
import kornia.feature as KF

KORNIA_AVAILABLE = True

class SimpleSatelliteMatcher:

    def __init__(self, device='cuda', weights_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        ssl._create_default_https_context = ssl._create_unverified_context

        if not KORNIA_AVAILABLE:
            self._init_sift()
            return

        try:
            print("Loading LoFTR...")
            self.matcher = KF.LoFTR(pretrained='outdoor').to(self.device)

            # load fine-tuned weights if provided
            if weights_path and Path(weights_path).exists():
                print(f"Loading fine-tuned weights: {weights_path}")
                self.matcher.load_state_dict(torch.load(weights_path))

            self.matcher.eval()
            self.use_loftr = True
            print("LoFTR is ready")
        except Exception as e:
            print(f"LoFTR failed: {e}")
            self._init_sift()

    def _init_sift(self):
        self.use_loftr = False
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        print("SIFT is ready")

    def match_images(self, img1_path, img2_path):
        # load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            raise ValueError("Failed to load images")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if self.use_loftr:
            return self._match_loftr(gray1, gray2, img1, img2)
        else:
            return self._match_sift(gray1, gray2, img1, img2)

    def _match_loftr(self, gray1, gray2, img1, img2):
        # resize for speed
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        max_dim = max(h1, w1, h2, w2)
        scale = min(1.0, 840 / max_dim)

        img1_resized = cv2.resize(gray1, None, fx=scale, fy=scale)
        img2_resized = cv2.resize(gray2, None, fx=scale, fy=scale)

        # convert to tensors
        img1_t = K.image_to_tensor(img1_resized, False).float() / 255.
        img2_t = K.image_to_tensor(img2_resized, False).float() / 255.
        img1_t = img1_t.to(self.device)
        img2_t = img2_t.to(self.device)

        with torch.no_grad():
            corr = self.matcher({'image0': img1_t, 'image1': img2_t})

        kpts0 = corr['keypoints0'].cpu().numpy() / scale
        kpts1 = corr['keypoints1'].cpu().numpy() / scale
        conf = corr['confidence'].cpu().numpy()

        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'confidence': conf,
            'num_matches': len(kpts0),
            'image0': img1,
            'image1': img2
        }

    def _match_sift(self, gray1, gray2, img1, img2):
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return {
                'keypoints0': np.zeros((0, 2)),
                'keypoints1': np.zeros((0, 2)),
                'confidence': np.zeros(0),
                'num_matches': 0,
                'image0': img1,
                'image1': img2
            }

        matches = self.matcher_bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) == 0:
            return {
                'keypoints0': np.zeros((0, 2)),
                'keypoints1': np.zeros((0, 2)),
                'confidence': np.zeros(0),
                'num_matches': 0,
                'image0': img1,
                'image1': img2
            }

        kpts0 = np.array([kp1[m.queryIdx].pt for m in good])
        kpts1 = np.array([kp2[m.trainIdx].pt for m in good])

        distances = np.array([m.distance for m in good])
        conf = 1 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'confidence': conf,
            'num_matches': len(kpts0),
            'image0': img1,
            'image1': img2
        }

    def visualize_matches(self, result, save_path=None, max_show=100):
        img1 = result['image0']
        img2 = result['image1']
        kpts1 = result['keypoints0']
        kpts2 = result['keypoints1']
        conf = result['confidence']

        if len(kpts1) == 0:
            print("No matches found")
            return None

        # create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        output = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1:] = img2

        # select top matches
        num_show = min(max_show, len(kpts1))
        if len(kpts1) > num_show:
            idx = np.argsort(-conf)[:num_show]
            kpts1 = kpts1[idx]
            kpts2 = kpts2[idx]
            conf = conf[idx]

        # draw matches
        for i in range(len(kpts1)):
            pt1 = tuple(kpts1[i].astype(int))
            pt2 = tuple((kpts2[i] + [w1, 0]).astype(int))
            color_val = int(255 * conf[i])
            color = (0, color_val, 255 - color_val)
            cv2.line(output, pt1, pt2, color, 1)
            cv2.circle(output, pt1, 2, color, -1)
            cv2.circle(output, pt2, 2, color, -1)


        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"Matches: {result['num_matches']} (showing {num_show}) | Confidence: {np.mean(result['confidence']):.3f}")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
        return output


def main():
    print("\n" + "="*50)
    print("Satellite Image Matching - Test")
    print("="*50 + "\n")

    # check for fine-tuned weights
    weights_path = 'loftr_finetuned_best.pth'
    if not Path(weights_path).exists():
        weights_path = None
        print("Using pretrained weights")
    else:
        print(f"Using fine-tuned weights: {weights_path}")

    matcher = SimpleSatelliteMatcher(weights_path=weights_path)

    # find test data
    test_data = Path(r"data\onera-dataset")
    if not test_data.exists():
        print(f"Error: Dataset not found at {test_data}")
        print("Update the path in the script")
        return

    cities = [d for d in test_data.iterdir()
              if d.is_dir() and (d / "pair").exists()]

    if not cities:
        print("No cities found")
        return

    # test first city
    city = cities[0]
    img1 = city / "pair" / "img1.png"
    img2 = city / "pair" / "img2.png"

    print(f"\nTesting: {city.name}")
    result = matcher.match_images(img1, img2)

    print(f"\nResults:")
    print(f"  Matches: {result['num_matches']}")
    if result['num_matches'] > 0:
        print(f"  Confidence: {np.mean(result['confidence']):.3f} ± {np.std(result['confidence']):.3f}")

    # visualize
    matcher.visualize_matches(result, f"test_result_{city.name}.png")

    print("\ndone!")


if __name__ == "__main__":
    main()