#!/usr/bin/env python3
"""
Standalone function to convert ASL citizen training data to Phoenix dataset format.

This script converts ASL CSV annotations with format:
['Participant ID', 'Video file', 'Gloss', 'ASL-LEX Code']

To Phoenix format:
['id', 'folder', 'signer', 'annotation']

Mapping:
- id = video file
- folder = video file
- signer = participant id
- annotation = gloss
"""

import pandas as pd
import sys
from pathlib import Path


def convert_asl_to_phoenix_format(asl_csv_path: str, output_path: str = None, output_format: str = 'csv'):
    """
    Convert ASL citizen training annotations to Phoenix dataset format.

    Args:
        asl_csv_path (str): Path to the ASL CSV file
        output_path (str, optional): Output file path. If None, auto-generates based on input filename
        output_format (str): Output format - 'xlsx', 'csv', or 'both'

    Returns:
        pd.DataFrame: Converted Phoenix format DataFrame
    """

    print("=" * 60)
    print("ASL TO PHOENIX FORMAT CONVERTER")
    print("=" * 60)

    # Step 1: Load ASL CSV file
    try:
        print(f"Loading ASL annotations from: {asl_csv_path}")
        asl_df = pd.read_csv(asl_csv_path)

        print(f"Original data shape: {asl_df.shape}")
        print(f"Original columns: {list(asl_df.columns)}")

        # Validate required columns
        required_columns = ['Participant ID', 'Video file', 'Gloss', 'ASL-LEX Code']
        missing_columns = [col for col in required_columns if col not in asl_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"✓ All required ASL columns found")

    except Exception as e:
        print(f"✗ Error loading ASL CSV: {e}")
        return None

    # Step 2: Convert to Phoenix format
    try:
        print("\nConverting to Phoenix format...")

        phoenix_df = pd.DataFrame()

        # Mapping as specified:
        # id = video file (remove .mp4 extension to match JSON filenames)
        phoenix_df['id'] = asl_df['Video file'].apply(lambda x: Path(str(x)).stem)

        # folder = video file (keep original for compatibility)
        phoenix_df['folder'] = asl_df['Video file'].astype(str)

        # signer = participant id
        phoenix_df['signer'] = asl_df['Participant ID'].astype(str)

        # annotation = gloss
        phoenix_df['annotation'] = asl_df['Gloss'].astype(str)

        # Clean up any rows with missing critical data
        initial_count = len(phoenix_df)
        phoenix_df = phoenix_df.dropna(subset=['id', 'annotation'])
        final_count = len(phoenix_df)

        if initial_count != final_count:
            print(f"⚠ Removed {initial_count - final_count} rows with missing data")

        print(f"✓ Conversion completed successfully")
        print(f"Phoenix data shape: {phoenix_df.shape}")
        print(f"Phoenix columns: {list(phoenix_df.columns)}")

    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return None

    # Step 3: Generate output filename if not provided
    if output_path is None:
        input_path = Path(asl_csv_path)
        base_name = input_path.stem + "_phoenix"
        output_dir = input_path.parent

        if output_format == 'csv':
            output_path = output_dir / f"{base_name}.csv"
        elif output_format == 'xlsx':
            output_path = output_dir / f"{base_name}.xlsx"
        else:  # both
            output_path = output_dir / base_name

    # Step 4: Save converted data
    try:
        print(f"\nSaving converted data...")

        if output_format == 'csv':
            phoenix_df.to_csv(output_path, index=False)
            print(f"✓ Saved CSV file: {output_path}")

        elif output_format == 'xlsx':
            phoenix_df.to_excel(output_path, index=False)
            print(f"✓ Saved Excel file: {output_path}")

        elif output_format == 'both':
            # Save both formats
            xlsx_path = str(output_path) + ".xlsx"
            csv_path = str(output_path) + ".csv"

            phoenix_df.to_excel(xlsx_path, index=False)
            phoenix_df.to_csv(csv_path, index=False)

            print(f"✓ Saved Excel file: {xlsx_path}")
            print(f"✓ Saved CSV file: {csv_path}")

    except Exception as e:
        print(f"✗ Error saving converted data: {e}")
        return phoenix_df  # Return DataFrame even if saving failed

    # Step 5: Display conversion summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    print(f"Input file: {asl_csv_path}")
    print(f"Original entries: {len(asl_df)}")
    print(f"Converted entries: {len(phoenix_df)}")
    print(f"Unique participants: {phoenix_df['signer'].nunique()}")
    print(f"Unique annotations: {phoenix_df['annotation'].nunique()}")

    # Show sample of converted data
    print(f"\nSample converted entries:")
    for i, row in phoenix_df.head(5).iterrows():
        print(f"  {i + 1}. ID: {row['id']}")
        print(f"     Signer: {row['signer']}")
        print(f"     Annotation: {row['annotation']}")
        print()

    print("✓ Conversion completed successfully!")
    print("=" * 60)

    return phoenix_df


def main():
    """Command line interface for the converter"""

    if len(sys.argv) < 2:
        print("Usage: python convert_asl_to_phoenix.py <asl_csv_file> [output_file] [format]")
        print("")
        print("Arguments:")
        print("  asl_csv_file    Path to ASL CSV file")
        print("  output_file     Output file path (optional)")
        print("  format          Output format: 'csv', 'xlsx', or 'both' (default: 'csv')")
        print("")
        print("Example:")
        print("  python convert_asl_to_phoenix.py asl_annotations.csv")
        print("  python convert_asl_to_phoenix.py asl_annotations.csv phoenix_annotations.csv")
        print("  python convert_asl_to_phoenix.py asl_annotations.csv phoenix_annotations both")
        sys.exit(1)

    asl_csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'csv'

    # Validate input file exists
    if not Path(asl_csv_path).exists():
        print(f"Error: Input file does not exist: {asl_csv_path}")
        sys.exit(1)

    # Validate output format
    if output_format not in ['csv', 'xlsx', 'both']:
        print(f"Error: Invalid output format '{output_format}'. Use 'csv', 'xlsx', or 'both'")
        sys.exit(1)

    # Run conversion
    result = convert_asl_to_phoenix_format(asl_csv_path, output_path, output_format)

    if result is not None:
        print(f"\n✓ Successfully converted {len(result)} entries!")
    else:
        print(f"\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()