#!/usr/bin/env python3
"""
Test script for inference pipeline.
Creates mock data and model, then runs end-to-end inference.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tempfile

# Add inference module to path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from pcap_ingestion import PCAPFlowExtractor
from feature_extraction import FeatureExtractor
from preprocessing import Preprocessor, NormalizationStats
from model_inference import ModelManager
from inference import InferencePipeline


def create_mock_model(model_path, num_classes=21):
    """Create a mock LSTM model for testing."""
    print(f"📦 Creating mock model at {model_path}...")
    
    # Input: 141 dimensions (128 sequence + 13 scalar)
    inputs = keras.Input(shape=(141,))
    
    # Simple LSTM-like network
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save model
    model.save(model_path)
    print(f"✅ Mock model saved to {model_path}")
    return model


def create_label_map(labels_path, num_classes=21):
    """Create a mock label mapping file."""
    print(f"📝 Creating label map at {labels_path}...")
    
    malware_families = [
        "Benign",
        "Trojan.Zeroaccess", "Trojan.Ebury", "Trojan.Rustock", "Trojan.Pushdo",
        "Worm.Kelihos", "Worm.Mirai", "Botnet.Conficker", "Botnet.Mydoom",
        "Ransomware.Cerber", "Ransomware.Cryptolocker", "Backdoor.Poison",
        "Rootkit.ZeroAccess", "Spyware.ZeuS", "Spyware.SpySheriff",
        "Trojan.Agent", "Adware.Yandex", "PUP.InstallCore",
        "Trojan.Packers", "Virus.Win32", "Virus.HTML"
    ][:num_classes]
    
    label_map = {str(i): label for i, label in enumerate(malware_families)}
    
    with open(labels_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"✅ Label map saved with {len(label_map)} classes")
    return label_map


def create_normalization_stats(stats_path):
    """Create mock normalization statistics."""
    print(f"📊 Creating normalization stats at {stats_path}...")
    
    # Create stats for 141 dimensions
    means = np.random.randn(141).tolist()
    stds = np.abs(np.random.randn(141)) + 0.5  # Ensure positive
    stds = stds.tolist()
    
    stats = {
        "means": means,
        "stds": stds,
        "feature_names": [
            # Sequence features (128)
            *[f"intervals_ccnt_{i}" for i in range(16)],
            *[f"pld_ccnt_{i}" for i in range(16)],
            *[f"rev_intervals_ccnt_{i}" for i in range(16)],
            *[f"rev_pld_ccnt_{i}" for i in range(16)],
            *[f"ack_psh_rst_syn_fin_cnt_{i}" for i in range(16)],
            *[f"rev_ack_psh_rst_syn_fin_cnt_{i}" for i in range(16)],
            *[f"extra_seq_{i}" for i in range(32)],  # Padding to 128
            # Scalar features (13)
            "pr", "rev_pld_max", "rev_pld_mean", "pld_mean", "pld_median",
            "pld_distinct", "time_length", "bytes_out", "bytes_in",
            "num_pkts_out", "num_pkts_in", "src_port", "dst_port"
        ]
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Normalization stats saved")
    return stats


def create_synthetic_pcap(pcap_path):
    """Create a synthetic PCAP file for testing."""
    print(f"🌐 Creating synthetic PCAP file at {pcap_path}...")
    
    try:
        from scapy.all import wrpcap, IP, TCP, UDP, Raw
        import time
        
        packets = []
        
        # Create a few sample flows
        # Flow 1: TCP traffic (Benign-like)
        for i in range(10):
            pkt = IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=443) / Raw(load=b"GET / HTTP/1.1\r\n")
            pkt.time = time.time() + i * 0.1
            packets.append(pkt)
        
        # Flow 2: UDP traffic (different ports)
        for i in range(8):
            pkt = IP(src="192.168.1.100", dst="1.1.1.1") / UDP(sport=53456, dport=53) / Raw(load=b"DNS query")
            pkt.time = time.time() + 2.0 + i * 0.15
            packets.append(pkt)
        
        # Flow 3: TCP traffic (reverse direction)
        for i in range(6):
            pkt = IP(src="10.0.0.50", dst="192.168.1.100") / TCP(sport=80, dport=54321) / Raw(load=b"HTTP response")
            pkt.time = time.time() + 4.0 + i * 0.2
            packets.append(pkt)
        
        wrpcap(pcap_path, packets)
        print(f"✅ Created synthetic PCAP with {len(packets)} packets")
        return True
        
    except ImportError:
        print("⚠️  Scapy not available, creating minimal PCAP manually...")
        # Create a minimal valid PCAP file
        import struct
        
        pcap_global_header = struct.pack('<4s2H3I4s', 
            b'\xa1\xb2\xc3\xd4',  # magic number
            2, 4,                  # version major, minor
            0, 0, 0,               # timezone, timestamp accuracy, snaplen
            1                      # network (Ethernet)
        )
        
        with open(pcap_path, 'wb') as f:
            f.write(pcap_global_header)
            
            # Write a few dummy packet headers
            for i in range(3):
                timestamp = 1702469800 + i
                payload = b"dummy packet data" + str(i).encode()
                packet_header = struct.pack('<4I',
                    timestamp,     # ts_sec
                    100000,        # ts_usec
                    len(payload),  # incl_len
                    len(payload)   # orig_len
                )
                f.write(packet_header)
                f.write(payload)
        
        print(f"✅ Created minimal PCAP file with dummy packets")
        return True


def main():
    """Run end-to-end inference test."""
    print("\n" + "="*70)
    print("🚀 INFERENCE PIPELINE TEST")
    print("="*70 + "\n")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # File paths
        model_path = tmpdir / "test_model.h5"
        labels_path = tmpdir / "labels.json"
        stats_path = tmpdir / "norm_stats.json"
        pcap_path = tmpdir / "test_traffic.pcap"
        output_dir = tmpdir / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Create test data
        print("\n[SETUP] Preparing test data...\n")
        create_mock_model(str(model_path), num_classes=21)
        create_label_map(str(labels_path), num_classes=21)
        create_normalization_stats(str(stats_path))
        create_synthetic_pcap(str(pcap_path))
        
        # Run inference
        print("\n[INFERENCE] Running pipeline...\n")
        pipeline = InferencePipeline(
            pcap_file=str(pcap_path),
            model_path=str(model_path),
            labels_path=str(labels_path),
            stats_path=str(stats_path),
            output_dir=str(output_dir),
            verbose=True
        )
        
        try:
            results = pipeline.run_pipeline()
            
            print("\n" + "="*70)
            print("✅ INFERENCE COMPLETED SUCCESSFULLY")
            print("="*70)
            
            if results:
                print(f"\n📊 Results Summary:")
                print(f"   - Total flows processed: {len(results)}")
                if len(results) > 0:
                    classes = [r.get('class_name', 'Unknown') for r in results]
                    unique_classes = set(classes)
                    print(f"   - Unique classes: {unique_classes}")
                    avg_confidence = np.mean([r.get('confidence', 0) for r in results])
                    print(f"   - Average confidence: {avg_confidence:.4f}")
            
            # Show output files
            print(f"\n📁 Output files generated:")
            for output_file in output_dir.glob("results.*"):
                size = output_file.stat().st_size
                print(f"   - {output_file.name} ({size} bytes)")
            
            # Display sample results
            json_file = output_dir / "results.json"
            if json_file.exists():
                print(f"\n📋 Sample Results (JSON):")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'predictions' in data and len(data['predictions']) > 0:
                        sample = data['predictions'][0]
                        print(f"   Flow 1:")
                        print(f"     - Class: {sample.get('class_name', 'N/A')}")
                        print(f"     - Confidence: {sample.get('confidence', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"\n❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✨ TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
