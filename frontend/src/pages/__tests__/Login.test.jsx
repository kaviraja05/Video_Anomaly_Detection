import React from 'react';
import { render, screen } from '@testing-library/react';
import Login from '../Login';

describe('Login Component', () => {
  it('renders login form properly', () => {
    // Basic test to ensure it mounts and contains expected form
    render(<Login onNavigate={() => {}} onLogin={() => {}} />);
    expect(screen.getByText('Welcome Back')).toBeTruthy();
  });
});
