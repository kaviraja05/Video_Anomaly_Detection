import React from 'react';
import { render, screen } from '@testing-library/react';
import UploadPage from '../UploadPage';

describe('UploadPage Component', () => {
  it('renders upload area properly', () => {
    render(<UploadPage onAnalysisComplete={() => {}} />);
    // Check for dropzone specific text
    expect(screen.getByText(/Upload Video|Analyze|Drag/i)).toBeTruthy();
  });
});
